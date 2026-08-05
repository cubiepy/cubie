"""
The main construct in bigmodelmanip is a :class:`bigmodelmanip.model.Model`.
This represents a flattened BigModel model and metadata about its variables.
"""
import logging
import numbers
from enum import Enum
from io import StringIO

import networkx as nx
import rdflib
import sympy

from .rdf import create_rdf_node
from .units import UnitStore

from . import parser


logger = logging.getLogger(__name__)

# Delimiter for variables name in Sympy expressions: <component><delimiter><name>
SYMPY_SYMBOL_DELIMITER = '$'

# Float precision to use when creating sympy.Float objects
FLOAT_PRECISION = 17


class Model(object):
    """
    A componentless representation of a BigModel model, containing a list of equations, units, and RDF metadata about
    variables used in those equations.

    The main parts of a Model are 1. a list of sympy equation objects; 2. a collection of named units; and 3. an RDF
    graph that stores further meta data about the model.

    Equations are stored as :class:`sympy.Eq` objects, but with the caveat that all variables and numbers must be
    specified using the :class:`sympy.Dummy` objects returned by :meth:`add_variable()` and :meth:`create_quantity()`.

    Units are handled using the ``units`` property of a model, which is an instance of
    :class:`bigmodelmanip.units.UnitStore`.

    RDF meta data can be attached to either the model itself or to model variables, via the BigModel ``cmeta:id``
    attribute. Cmeta ids set on any other parts of BigModel models are ignored.

    Bigmodelmanip does not support algebraic models: the left-hand side every equation in the model must be a variable or
    a derivative.

    :param name: the name of the model e.g. from ``<model name="">``.
    :param cmeta_id: An optional cmeta id, e.g. from ``<model cmeta:id="">``.
    :param unit_store: Optional :class:`bigmodelmanip.units.UnitStore` instance; if given the model will share the
        underlying registry so that conversions between model units and those from the provided store work.
    """

    def __init__(self, name, cmeta_id=None, unit_store=None):

        self.name = name
        self._cmeta_id = cmeta_id
        self.rdf_identity = create_rdf_node('#' + cmeta_id) if cmeta_id else None

        # A list of sympy.Eq equation objects
        self.equations = []

        # A UnitStore object
        if unit_store:
            self.units = UnitStore(unit_store)
        else:
            self.units = UnitStore()

        # Maps string variable names to Variable objects
        self._name_to_variable = {}

        # Maps cmeta ids to Variable objects
        self._cmeta_id_to_variable = {}

        # Cached nx.DiGraph of this model's equations, with number dummies or with sympy.Number objects
        self._graph = None
        self._graph_with_sympy_numbers = None

        # An RDF graph containing further meta data
        self.rdf = rdflib.Graph()

        # Map from Variable to defining equation, where the variable is defined by a simple equation
        self._var_definition_map = {}

        # Map from Variable to defining equation, where the variable is defined by an ODE
        self._ode_definition_map = {}

    ####################################################################################################
    # Main methods to query information from the model

    def variables(self):
        """Returns an iterator over this model's variables."""
        return self._name_to_variable.values()

    def get_free_variable(self):
        """Returns the free variable in this model (if any)."""
        for ode in self._ode_definition_map.values():
            free_variable = ode.lhs.variables[0]
            return free_variable

        raise ValueError('No free variable set in model.')

    def get_state_variables(self, sort=True):
        """
        Returns a list of state variables found in the given model graph (ordered by appearance in the BigModel document).

        :param sort: indicates whether the list is sorted by appearance in the BigModel document.
        """
        states = list(self._ode_definition_map.keys())
        if sort:
            states.sort(key=lambda state_var: state_var.order_added)
        return states

    def get_derivatives(self, sort=True):
        """Returns a list of :class:`sympy.Derivative` objects found as LHS in the given model graph.

        :param sort: indicates whether the list is sorted by appearance in the BigModel document.
        """
        derivatives = [v for v in self.graph if isinstance(v, sympy.Derivative)]
        if sort:
            derivatives.sort(key=lambda deriv: deriv.args[0].order_added)
        return derivatives

    def get_derived_quantities(self, sort=True):
        """Returns a list of derived quantities found in the given model graph.

        A derived quantity is any variable that is not a state variable, free variable, or parameter/constant.
        :param sort: indicates whether the list is sorted by appearance in the BigModel document.
        """
        derived_quantities = [
            v for v, node in self.graph.nodes.items()
            if not isinstance(v, sympy.Derivative)
            and node.get('variable_type', VariableType.UNKNOWN) not in (
                VariableType.FREE, VariableType.STATE, VariableType.PARAMETER)]
        if sort:
            derived_quantities.sort(key=lambda var: var.order_added)
        return derived_quantities

    def get_display_name(self, var, ontology=None, exclude_terms=None):
        """Return a display name for the given variable.

        Looks for an annotation in the ontology first (or the local name from any annotation if no ontology is
        specified), skipping any terms in exclude_tags,
        then ``cmeta:id`` if present, or the variable's ``name`` attribute if not.

        Dollar symbols in the name are replaced by a double underscore.

        :param var: the variable for which to get the display name.
        :param ontology: the base URL of an ontology if only annotations within that ontology should be considered
        :param exclude_terms: a collection of terms
               that should be skipped when searching the ontology for a variable name.

        :return: the display name for the variable according to the algorithm above
        """
        if self.has_ontology_annotation(var, ontology):
            for term in reversed(self.get_ontology_terms_by_variable(var, ontology)):
                if exclude_terms is None or term not in exclude_terms:
                    return term
        return var.cmeta_id if var.cmeta_id else var.name.replace('$', '__')

    def is_state(self, variable):
        """Checks if ``variable`` is a state variable (i.e. if it's defined by an ODE)."""
        return variable in self._ode_definition_map

    def is_constant(self, variable):
        """Determine whether ``variable`` is a constant.

        This is calculated by looking at the RHS of the defining equation and checking it has no
        variable references.
        """
        defn = self._var_definition_map.get(variable)
        return defn is not None and len(defn.rhs.atoms(Variable)) == 0

    def get_variable_by_name(self, name):
        """Returns the variable with the given ``name``."""
        return self._name_to_variable[name]

    def get_variable_by_cmeta_id(self, cmeta_id):
        """
        Searches the model and returns the variable with the given ``cmeta id``.

        To get variables from e.g. an annotation ontology term, use :meth:`get_variable_by_ontology_term`.

        :param cmeta_id: Either a string id or :class:`rdflib.URIRef` instance.
        :returns: A :class:`Variable` object
        """

        # Get cmeta_id from URIRef
        if isinstance(cmeta_id, rdflib.term.Node):
            assert isinstance(cmeta_id, rdflib.URIRef), 'Non-resource {} annotated.'.format(cmeta_id)
            cmeta_id = str(cmeta_id)
            if cmeta_id[0] != '#':
                # TODO This should eventually be implemented?
                raise NotImplementedError(
                    'Non-local annotations are not supported.')
            cmeta_id = cmeta_id[1:]

        # Get variable by cmeta id string
        try:
            return self._cmeta_id_to_variable[cmeta_id]
        except KeyError:
            raise KeyError('No variable with cmeta id "%s" found.' % str(cmeta_id))

    def get_variable_by_ontology_term(self, term):
        """Searches the RDF graph for a variable annotated with the given ``term`` and returns it.

        Specifically, this method searches for a unique variable annotated with predicate
        ``http://www.bigmodel.org/qualifiers/is`` and the object specified by ``term``.

        Will raise a ``KeyError`` if no variable with the given annotation is found, and a ``ValueError`` if more than
        one variable with the given annotation is found.

        :param term: anything suitable as an input to :meth:`bigmodelmanip.rdf.create_rdf_node`; typically either an RDF
            node already, or a tuple ``(namespace_uri, local_name)``.
        """
        variables = self.get_variables_by_rdf(('http://www.bigmodel.org/qualifiers/', 'is'), term)
        if len(variables) == 1:
            return variables[0]
        elif not variables:
            raise KeyError('No variable annotated with {} found.'.format(term))
        else:
            raise ValueError('Multiple variables annotated with {}'.format(term))

    def get_variables_by_rdf(self, predicate, object_=None, sort=True):
        """Find variables annotated with the given predicate and object (e.g. ``is annotation:time``) in our RDF graph.

        Both ``predicate`` and ``object_`` (if given) must be suitable as an input to
        :meth:`bigmodelmanip.rdf.create_rdf_node`; typically either ``(namespace, local_name)`` tuples or string literals.

        :param sort: indicates whether the list is sorted by appearance in the BigModel document.

        :return: the associated variables sorted in document order
        """
        predicate = create_rdf_node(predicate)
        object_ = create_rdf_node(object_)

        # Find variables, sort and return
        variables = [self.get_variable_by_cmeta_id(result) for result in self.rdf.subjects(predicate, object_)]
        if sort:
            variables.sort(key=lambda sym: sym.order_added)
        return variables

    def get_rdf_value(self, subject, predicate):
        """Get the value of an RDF object connected to ``subject`` by ``predicate``.

        Note: expects exactly one triple to match and the result to be a literal.

        :param subject: the object of the triple returned
        :param predicate: the object of the triple returned
        :returns: a string
        """
        triples = self.get_rdf_annotations(subject, predicate)
        triples = tuple(self.get_rdf_annotations(subject, predicate))
        assert len(triples) == 1, 'Expecting exactly 1 triple'
        triple = triples[0]
        assert isinstance(triple[2], rdflib.Literal)
        value = str(triple[2]).strip()  # Could make this cleverer by considering data type if desired
        return value

    def get_rdf_annotations(self, subject=None, predicate=None, object_=None):
        """Searches the RDF graph and returns triples matching the given parameters.

        :param subject: the subject of the triples returned
        :param predicate: the predicate of the triples returned
        :param object_: the object of the triples returned

        Each of ``subject``, ``predicate`` and ``object_`` are optional; if ``None`` then any triple matches.
        If all are ``None``, then all triples are returned.

        The arguments can be anything valid as input to :meth:`bigmodelmanip.rdf.create_rdf_node`,
        typically a (namespace URI, local name) pair, a string, or ``None``.
        """
        subject = create_rdf_node(subject)
        predicate = create_rdf_node(predicate)
        object_ = create_rdf_node(object_)
        return self.rdf.triples((subject, predicate, object_))

    def get_ontology_terms_by_variable(self, variable, namespace_uri=None):
        """
        Returns all ontology terms linked to the given ``variable`` via the
        ``http://www.bigmodel.org/qualifiers/is`` predicate.

        :param variable: The variable to search for (as a :class:`Variable` object).
        :param namespace_uri: An optional namespace URI. If given, only terms within the given namespace will be
            returned.
        :returns: A list of term local names.
        """
        ontology_terms = []
        if variable.rdf_identity:
            predicate = create_rdf_node(('http://www.bigmodel.org/qualifiers/', 'is'))
            for object in self.rdf.objects(variable.rdf_identity, predicate):
                # Filter by namespace
                if namespace_uri is None or str(object).startswith(namespace_uri):
                    uri_parts = str(object).split('#')
                    ontology_terms.append(uri_parts[-1])
        return ontology_terms

    def has_ontology_annotation(self, variable, namespace_uri=None):
        """
        Checks that there is at least one result for :meth:`Model.get_ontology_terms_by_variable` with the given
        arguments.
        """
        return len(self.get_ontology_terms_by_variable(variable, namespace_uri)) != 0

    def has_cmeta_id(self, cmeta_id):
        """
        Returns ``True`` only if the given ``cmeta_id`` exists in this model.

        Note that only cmeta ids on variables or the model itself are checked and supported.
        """
        # Check if it's the model id
        if cmeta_id == self._cmeta_id and cmeta_id is not None:
            return True

        # Check if it's a variable id. All other cmeta_ids in the original BigModel are ignored.
        return cmeta_id in self._cmeta_id_to_variable

    def get_definition(self, variable):
        """Get the equation (if any) defining the given variable.

        :param variable: The variable to look up (as a :class:`Variable`). If this appears as the LHS of a straight
            assignment, or the state variable in an ODE, the corresponding equation will be returned.
        :returns: A Sympy equation, or ``None`` if the variable is not defined by an equation.
        """
        defn = self._ode_definition_map.get(variable)
        if defn is None:
            defn = self._var_definition_map.get(variable)
        return defn

    def get_value(self, variable):
        """
        Returns the evaluated value of the given variable, as a float.

        For state variables, this returns the initial value. For variables that depend on other variables this
        recursively evaluates any dependencies (again at the initial state). Zero is returned for the free variable (as
        identified by :meth:`get_free_variable()`); trying to evaluate other variables without a definition will
        result in a ``ValueError``.
        """
        return self._get_value(variable)

    def _get_value(self, variable, evaluated=None):
        """Internal method to implement :meth:`get_value()`."""

        # State? Then return initial value
        if variable in self._ode_definition_map:
            return float(variable.initial_value)

        # Get RHS and evaluate
        try:
            expr = self._var_definition_map[variable]
        except KeyError:
            if self._ode_definition_map and variable is self.get_free_variable():
                return 0
            raise ValueError('No definition set for ' + self.get_display_name(variable))
        expr = expr.rhs
        deps = expr.atoms(Variable)
        if deps:
            if evaluated is None:
                evaluated = {x: x.initial_value for x in self._ode_definition_map.keys()}
                if self._ode_definition_map:
                    time = self.get_free_variable()
                    evaluated[time] = 0
            for dep in deps:
                if dep not in evaluated:
                    evaluated[dep] = self._get_value(dep, evaluated)
            expr = expr.xreplace(evaluated)
            deps = expr.atoms(Variable)

        return float(expr)

    def get_equations_for(self, variables, recurse=True, strip_units=True):
        """Get all equations for a given collection of variables.

        Results are sorted first by dependencies, then by variable name.

        :param variables: The variables to get the equations for (as :class:`Variable` objects).
        :param recurse: Indicates whether to recurse the equation graph, or to return only the top level equations.
        :param strip_units: If ``True``, all :class:`Quantity` objects representing number with units will be
            replaced with ordinary sympy number objects. Note that if this is done then the equations returned may *not*
            be found in the model, so you should not try calling e.g. :meth:`remove_equation` with them.
        """
        # Get graph
        if strip_units:
            graph = self.graph_with_sympy_numbers
        else:
            graph = self.graph

        # Get sorted list of variables
        sorted_variables = nx.lexicographical_topological_sort(graph, key=str)

        # Create set of variables for which we require equations
        required_variables = set()
        for output in variables:
            required_variables.add(output)
            if recurse:
                required_variables.update(nx.ancestors(graph, output))
            else:
                required_variables.update(graph.pred[output])

        eqs = []
        for variable in sorted_variables:
            # Ignore variables we don't need
            if variable not in required_variables:
                continue

            # Get equation
            eq = graph.nodes[variable]['equation']

            # Skip variables that are not set with an equation
            if eq is None:
                continue

            eqs.append(eq)

        return eqs

    @property
    def graph(self):
        """A :class:`networkx.DiGraph` containing the model equations."""

        # Return cached graph
        if self._graph is not None:
            return self._graph

        # Store variables, their attributes and their relationships in a directed graph
        graph = nx.DiGraph()

        equation_count = 0

        # Add a node for every variable in the model, and set variable types
        for equation in self.equations:
            equation_count += 1

            # Add the lhs of the equation to the graph
            lhs = equation.lhs
            graph.add_node(lhs, equation=equation)

            # Update variable meta data based on the variable's role in the model
            if lhs.is_Derivative:
                # Get the state symbol and update the variable information
                state_symbol = lhs.free_symbols.pop()
                state_symbol.type = VariableType.STATE

                # Get the free symbol and update the variable information
                free_symbol = lhs.variables[0]
                free_symbol.type = VariableType.FREE
            elif isinstance(equation.rhs, Quantity):
                lhs.type = VariableType.PARAMETER
            else:
                lhs.type = VariableType.COMPUTED

        # Sanity check: none of the lhs have the same hash
        assert len(graph.nodes) == equation_count

        # Sanity check: all the lhs are unique in meaning (sympy.Dummy: same name != same hash)
        assert len(set([str(x) for x in graph.nodes])) == equation_count

        # Add edges between the nodes
        for equation in self.equations:
            lhs = equation.lhs

            # for each of the symbols or derivatives on the rhs of the equation
            for rhs in self.find_variables_and_derivatives([equation.rhs]):

                if rhs in graph.nodes:
                    # If the symbol maps to a node in the graph just add the dependency edge
                    graph.add_edge(rhs, lhs)
                elif rhs.type in [VariableType.STATE, VariableType.FREE]:
                    # If the variable is a state or free variable of a derivative
                    graph.add_node(rhs, equation=None, variable_type=rhs.type)
                    graph.add_edge(rhs, lhs)
                else:
                    assert False, 'Unexpected variable {} on RHS'.format(rhs)  # pragma: no cover

            # check that the free  and state variables are defined as a node
            if lhs.is_Derivative:
                state_symbol = lhs.free_symbols.pop()
                free_symbol = lhs.variables[0]
                if free_symbol not in graph.nodes:
                    graph.add_node(free_symbol, equation=None, variable_type=free_symbol.type)
                if state_symbol not in graph.nodes:
                    graph.add_node(state_symbol, equation=None, variable_type=state_symbol.type)

        # Store variable type in the graph too
        for variable in graph.nodes:
            if not variable.is_Derivative:
                assert variable.type is not None
                graph.nodes[variable]['variable_type'] = variable.type

        # Cache graph and return
        self._graph = graph
        return graph

    @property
    def graph_with_sympy_numbers(self):
        """
        A :class:`networkx.DiGraph` containing the model equations, but with numbers represented as
        :class:`sympy.Number` objects instead of :class:`Quantity` objects.
        """
        if self._graph_with_sympy_numbers is not None:
            return self._graph_with_sympy_numbers

        # Get a clone of the graph
        graph = self.graph.copy()

        # Replace dummies with Float objects
        for node in graph.nodes:
            equation = graph.nodes[node]['equation']
            if equation is None:
                continue

            # Get all the dummy numbers on the RHS, and prepare substitution map
            dummies = equation.rhs.atoms(Quantity)
            subs_dict = {d: d.evalf(FLOAT_PRECISION) for d in dummies}

            # And replace the equation with one with the rhs subbed with sympy.Number objects
            if subs_dict:
                # Update rhs
                rhs = equation.rhs.xreplace(subs_dict)

                # Check if simplification removed dependencies on other variables, and if so remove the corresponding
                # edges.
                refs = self.find_variables_and_derivatives([rhs])
                edges = tuple(graph.in_edges(equation.lhs))
                for edge in edges:
                    ref = edge[0]
                    if ref not in refs:
                        graph.remove_edge(ref, equation.lhs)

                # Replace equation
                graph.nodes[node]['equation'] = sympy.Eq(equation.lhs, rhs)

        # Cache graph and return
        self._graph_with_sympy_numbers = graph
        return graph

    def get_unique_name(self, name):
        """
        Creates and returns a unique variable name, not used in the model.

        :param str name: Suggested unique name.
        :return str: Guaranteed unique name.
        """
        if name in self._name_to_variable:
            name = self.get_unique_name(name + '_a')
        return name

    ####################################################################################################
    # Model manipulation methods

    def add_variable(self, name, units, initial_value=None, public_interface=None, private_interface=None,
                     cmeta_id=None):
        """
        Adds a variable to the model and returns a :class:`Variable` to represent it in sympy expressions.

        :param name: A string name.
        :param units: A string unit name or a :class:`~bigmodelmanip.units.UnitStore.Unit` object.
        :param initial_value: An optional initial value.
        :param public_interface: An optional public interface specifier (only required when parsing BigModel).
        :param private_interface: An optional private interface specifier (only required when parsing BigModel).
        :param cmeta_id: An optional string specifying a cmeta id
        :raises ValueError: If a variable with that name already exists, or the given cmeta id is already taken.
        :return: A :class:`Variable` object.
        """
        # Check for clashes
        if name in self._name_to_variable:
            raise ValueError('Variable %s already exists.' % name)

        # Check uniqueness of cmeta id
        if cmeta_id is not None and self.has_cmeta_id(cmeta_id):
            raise ValueError('The cmeta id "%s" is already in use.' % cmeta_id)

        # Check units
        if not isinstance(units, self.units.Unit):
            units = self.units.get_unit(units)

        # Add variable
        self._name_to_variable[name] = var = Variable(
            name=name,
            units=units,
            model=self,
            initial_value=initial_value,
            public_interface=public_interface,
            private_interface=private_interface,
            order_added=len(self._name_to_variable),
            cmeta_id=cmeta_id,
        )

        # Add cmeta id to var mapping
        if cmeta_id is not None:
            self._cmeta_id_to_variable[cmeta_id] = var

        # Invalidate cached graphs
        self._invalidate_cache()

        return var

    def remove_variable(self, variable):
        """Remove a variable and its defining equation from the model.

        This will remove the equation either that defines ``variable`` directly, or if it is a state variable the
        corresponding ODE. All annotations about this variable are also removed from the RDF graph.

        :param Variable variable: the variable to remove
        """
        # Remove defining equation
        defn = self.get_definition(variable)
        if defn is not None:
            self.remove_equation(defn)

        # Remove any annotations
        if variable.rdf_identity:
            for triple in self.rdf.triples((variable.rdf_identity, None, None)):
                self.rdf.remove(triple)

        # Remove references to variable and invalidate cache
        del self._name_to_variable[variable.name]
        if variable._cmeta_id is not None:
            del self._cmeta_id_to_variable[variable._cmeta_id]
        variable._model = None  # Just in case!
        self._invalidate_cache()

    def add_equation(self, equation, check_duplicates=True):
        """
        Adds an equation to this model.

        The left-hand side (LHS) of the equation must be either a variable (as a :class:`Variable`) or a derivative
        (as a :class:`sympy.Derivative`).

        All numbers and variables used in the equation must have been obtained from this model, e.g. via
        :meth:`create_quantity()`, :meth:`add_variable()`, or :meth:`get_variable_by_ontology_term()`.

        :param equation: A :class:`sympy.Eq` object.
        :param check_duplicates: whether to check that the equation's LHS is not already defined
        """
        assert isinstance(equation, sympy.Eq), 'The argument `equation` must be a sympy.Eq.'
        lhs = equation.lhs
        if lhs.is_Derivative:
            if len(lhs.args) > 2 or lhs.args[1][1] > 1:
                raise ValueError('Only first order derivatives wrt a single variable are supported')
        self.equations.append(equation)
        if lhs.is_Derivative:
            state_var = lhs.free_symbols.pop()
            if check_duplicates:
                self._check_duplicate_definitions(state_var, equation)
            self._ode_definition_map[state_var] = equation
        elif isinstance(lhs, Variable):
            if check_duplicates:
                self._check_duplicate_definitions(lhs, equation)
            self._var_definition_map[lhs] = equation
        else:
            raise ValueError('Equation LHS should be a derivative or variable, not {}'.format(lhs))
        self._invalidate_cache()

    def remove_equation(self, equation):
        """
        Removes an equation from the model.

        :param equation: The equation to remove.
        """
        try:
            self.equations.remove(equation)
        except ValueError:
            raise KeyError('Equation not found in model ' + str(equation))

        # Update dependency maps
        lhs = equation.lhs
        if lhs.is_Derivative:
            del self._ode_definition_map[lhs.free_symbols.pop()]
        else:
            del self._var_definition_map[lhs]

        # Invalidate cached equation graphs
        self._invalidate_cache()

    def create_quantity(self, value, units):
        """
        Creates and returns a :class:`Quantity` to represent a number with units in Sympy expressions.

        Use this method rather than creating quantities directly to ensure their units are compatible with those used by
        the model, so unit conversion etc. works.

        :param number: A number (anything convertible to float).
        :param units: A string unit name or a :class:`~bigmodelmanip.units.UnitStore.Unit` object.

        :return: A :class:`Quantity` object.
        """

        # Check units
        if not isinstance(units, self.units.Unit):
            units = self.units.get_unit(units)

        return Quantity(value, units)

    def add_cmeta_id(self, variable):
        """
        Adds a (unique) cmeta id to the given variable.

        If the variable already has a cmeta id no action is performed.

        :param variable: A :class:`Variable`.
        """
        if variable._cmeta_id is not None:
            return

        # Create new cmeta id
        cmeta_id = self.get_display_name(variable)
        while self.has_cmeta_id(cmeta_id):
            cmeta_id += '_'

        # Add to variable and store in mapping
        variable._set_cmeta_id(cmeta_id)
        self._cmeta_id_to_variable[cmeta_id] = variable

    def add_rdf(self, rdf):
        """Takes an RDF string and stores it in the model's RDF graph."""
        self.rdf.parse(StringIO(rdf), format='xml')

    def transfer_cmeta_id(self, source, target):
        """
        Removes the ``cmeta_id`` from the variable ``source`` and adds it to ``target``.

        Raises a ``ValueError`` if ``source`` doesn't have a cmeta id, or if ``target`` already has a cmeta id.
        """
        if source._cmeta_id is None:
            raise ValueError('Cannot transfer cmeta id: source variable has no cmeta id.')
        if target._cmeta_id is not None:
            raise ValueError('Cannot transfer cmeta id: target variable already has a cmeta id.')

        # Transfer id
        target._set_cmeta_id(source._cmeta_id)
        source._set_cmeta_id(None)

        # Update mapping
        self._cmeta_id_to_variable[target._cmeta_id] = target

    def convert_variable(self, original_variable, units, direction, move_annotations=True):
        """
        Ensures the model contains a variable representing ``original_variable`` in the specified ``units``.

        If the variable is already in the required units, nothing happens, and ``original_variable`` is returned.

        If the variable's units can be converted to the new ``units``, a new variable will created in these units, and
        the ``cmeta:id`` attribute of ``original_variable`` will be moved to the new variable, so that all annotations
        are transferred to the new variable (unless ``move_annotations`` is given as ``False``).

        The ``direction`` argument specifies how information flows between the new variable and the original, and hence
        what new equation(s) will be added to the model to perform the conversion.
        If ``direction`` is ``DataDirectionFlow.INPUT``, then the original variable takes its value from the newly added
        variable; if it is ``DataDirectionFlow.OUTPUT`` then the opposite happens.
        If the direction is ``INPUT`` then any initial value will be moved to the new variable (and converted
        appropriately).

        For example, a model::

            var time :: ms {cmeta_id: time}
            var sv1 :: mV {cmeta_id: sv11, init: 2}

            ode(sv1, time) = 1 :: mV_per_ms

        transformed with::

            convert_variable(sv11, volt, DataDirectionFlow.OUTPUT)

        becomes::

            var time :: ms {cmeta_id: time}
            var sv1 :: mV {init: 2}
            var sv1_converted :: volt {cmeta_id: sv11}

            ode(sv1, time) = 1 :: mV_per_ms
            sv1_converted = sv1 * 0.001 :: V_per_mV

        If the information flow is reversed, i.e. with::

            convert_variable(sv11, volt, DataDirectionFlow.INPUT)

        then the model becomes::

            var time :: ms {cmeta_id: time}
            var sv1 :: mV
            var sv1_converted :: volt {cmeta_id: sv11, init: 0.002}

            ode(sv1_converted, time) = (1 :: mV_per_ms) * 0.001 :: V_per_mV
            sv1 = sv1_converted * 1000 :: mV_per_V

        Converting time as an input requires further processing, because every ODE needs adapting. With::

            convert_variable(time, second, DataDirectionFlow.INPUT)

        the model becomes::

            var time :: ms
            var time_converted :: s {cmeta_id: time}
            var sv1 :: mV {cmeta_id: sv11, init: 2}
            var sv1_orig_deriv :: mV_per_ms

            time = 1000 :: ms_per_s * time_converted
            sv1_orig_deriv = 1 :: mV_per_ms
            ode(sv1, time_converted) = 1000 :: ms_per_s * sv1_orig_deriv

        :param original_variable: the :class:`Variable` object representing the variable in the model to be
                                  converted
        :param units: a :class:`~bigmodelmanip.units.UnitStore.Unit` object representing the units to convert variable to
                      (note if variable is already in these units, model remains unchanged and the original variable is
                      returned)
        :param direction: either DataDirectionFlow.INPUT: the variable to be changed is an input and all affected
                          equations will be adjusted;
                          or DataDirectionFlow.OUTPUT: the variable to be changed is an output, equations
                          are unaffected apart from converting the actual output
        :param move_annotations: whether to point metadata annotations at the converted variable instead of the original
        :return: new variable with desired units, or original unchanged if conversion was not necessary
        :raises DimensionalityError: if the unit conversion is impossible
        """
        # Sanity checks on inputs
        assert isinstance(original_variable, Variable)
        assert original_variable.name in self._name_to_variable  # Variable must be in model
        assert isinstance(units, self.units.Unit)  # Units must be in the right registry
        assert isinstance(direction, DataDirectionFlow)

        # Compute conversion factor for old units to new;
        # throws DimensionalityError if unit conversion is not possible
        cf = self.units.get_conversion_factor(from_unit=original_variable.units, to_unit=units)
        if cf == 1:
            # No conversion necessary. The method above will ensure a factor close to 1 is returned as 1.
            return original_variable

        if isinstance(cf, numbers.Number):
            # Make the conversion factor a number symbol with explicit units
            cf = self.create_quantity(cf, units / original_variable.units)

        # Store original state and free symbols (these might change, so need to store references early)
        state_symbols = self.get_state_variables()
        try:
            free_symbol = self.get_free_variable()
        except ValueError:
            # No free variable, so don't need to worry about ODE conversion
            free_symbol = None

        # Create new variable and equations defining it and/or the original variable
        new_variable = self._convert_variable_instance(original_variable, cf, units, direction, move_annotations)

        # For outputs do not need to do additional changes for state/free symbols, so we're done
        if direction == DataDirectionFlow.OUTPUT:
            return new_variable

        derivative_replacements = {}  # A map from old derivatives to variables holding original RHS definitions

        if original_variable in state_symbols:
            # Make the converted variable the new state variable
            derivative_replacements.update(self._convert_state_variable_deriv(original_variable, new_variable, cf))

        if original_variable == free_symbol:
            # Change every ODE to be w.r.t. the new time variable
            # Process ODEs in model order to ensure new equations are added in a consistent order
            for _, ode in sorted(self._ode_definition_map.items(), key=lambda v_eq: v_eq[0].order_added):
                assert ode.args[0].args[1].args[0] == original_variable, "Can only have 1 free variable in the model"
                derivative_replacements.update(self._convert_free_variable_deriv(ode, new_variable, cf))

        # Replace any instances of derivatives of the RHS of other equations with variables holding the original
        # definitions of those derivatives
        if derivative_replacements:
            self._replace_references_to_derivatives(derivative_replacements)

        self._invalidate_cache()

        return new_variable

    ####################################################################################################
    # Internal helper methods

    def find_variables_and_derivatives(self, expressions):
        """Returns a set containing all variables and derivatives referenced in a list of expressions.

        Note that we can't just use ``.atoms(Variable, sympy.Derivative)`` for this, because it
        will return the state and free variables from inside derivatives, which is not what we want.

        :param expressions: an iterable of expressions to get variables for.
        :return: a set of variables and derivatives, as :class:`Variable` and :class:`sympy.Derivative`
            objects respectively.
        """
        variables = set()
        for expr in expressions:
            if expr.is_Derivative or isinstance(expr, Variable):
                variables.add(expr)
            else:
                variables |= self.find_variables_and_derivatives(expr.args)
        return variables

    def _invalidate_cache(self):
        """Removes cached graphs: should be called after manipulating variables or equations."""
        self._graph = None
        self._graph_with_sympy_numbers = None

    def _check_duplicate_definitions(self, var, equation):
        """
        Assert that a variable doesn't have an existing definition.

        :param var: the :class:`Variable`, either a state var or normal var
        :param equation: the new definition being added
        """
        if var in self._ode_definition_map:
            raise ValueError('The variable {} is defined twice ({} and {})'.format(
                var, equation, self._ode_definition_map[var]))
        if var in self._var_definition_map:
            raise ValueError('The variable {} is defined twice ({} and {})'.format(
                var, equation, self._var_definition_map[var]))

    def _replace_references_to_derivatives(self, derivative_replacement_map):
        """
        Replace all references to pre-conversion derivatives on the RHS of model equations.

        :param derivative_replacement_map: a map from old :class:`sympy.Derivative` expressions to their unit-converted
            replacements
        """
        derivatives_to_replace = set(derivative_replacement_map.keys())
        for equation in self.equations.copy():
            if not derivatives_to_replace.isdisjoint(equation.rhs.atoms(sympy.Derivative)):
                self.remove_equation(equation)
                self.add_equation(equation.xreplace(derivative_replacement_map))

    def _remove_ode_and_assign_rhs_to_new_variable(self, original_ode, original_state_variable):
        """
        Create a new variable holding the original RHS for an ODE, with an equation assigning it.

        Also removes the original ODE from the model. A replacement will be created by the caller.

        :param original_ode: the original derivative equation
        :param original_state_variable: the dependent variable
        :return: the new variable for the right hand side of the original ODE
        """
        # Create a variable to hold the original RHS
        deriv_name = self.get_unique_name(original_state_variable.name + '_orig_deriv')
        deriv_units = self.units.evaluate_units(original_ode.lhs)
        rhs_variable = self.add_variable(name=deriv_name, units=deriv_units)

        # Create new equation and remove original ODE
        expression = sympy.Eq(rhs_variable, original_ode.rhs)
        self.remove_equation(original_ode)
        self.add_equation(expression)

        return rhs_variable

    def _convert_free_variable_deriv(self, original_ode, new_time, cf):
        """
        Create relevant variables/equations when converting a free variable within a single ODE.

        See :meth:`convert_variable` for an example of how this works.

        :param original_ode: the derivative equation containing the pre-conversion free variable
        :param new_time: the new variable representing the converted free variable [new_units]
        :param cf: conversion factor for unit conversion [new units/old units]
        :return: a map from the old derivative term to the variable holding its original RHS
        """
        state_variable = original_ode.lhs.args[0]  # units [x]
        # Create a variable to hold the value of the original RHS, and the equation assigning it.
        # Will have units [x/old units]
        original_rhs_variable = self._remove_ode_and_assign_rhs_to_new_variable(original_ode, state_variable)

        # Add equation for derivative wrt new variable
        # dx/dnewvar [x/new units] = original_rhs_variable [x/old units] / cf [new units/old units]
        new_ode = sympy.Eq(sympy.Derivative(state_variable, new_time), original_rhs_variable / cf)
        self.add_equation(new_ode)
        return {original_ode.lhs: original_rhs_variable}

    def _convert_state_variable_deriv(self, original_variable, new_variable, cf):
        """
        Create relevant variables/equations when converting a state variable as an input.

        See :meth:`convert_variable` for an example of how this works.

        :param original_variable: the state variable to be converted [old units]
        :param new_variable: the new variable representing the converted symbol [new units]
        :param cf: conversion factor for unit conversion [new units/old units]
        :return: a map from the old derivative term to the variable holding its original RHS
        """
        original_ode = self._ode_definition_map[original_variable]
        free_variable = original_ode.lhs.args[1]  # units [t]

        # Create a variable to hold the value of the original RHS, and the equation assigning it.
        # Will have units [old units/t]
        original_rhs_variable = self._remove_ode_and_assign_rhs_to_new_variable(original_ode, original_variable)

        # Add the new ODE
        # dnewvar/dt [new units/t] = original_rhs_variable [old units/t] * cf [new units/old units]
        new_ode = sympy.Eq(sympy.Derivative(new_variable, free_variable), original_rhs_variable * cf)
        self.add_equation(new_ode)
        return {original_ode.lhs: original_rhs_variable}

    def _convert_variable_instance(self, original_variable, cf, units, direction, move_annotations):
        """
        Internal function to create new variable in given units, possibly with defining equation.

        The defining equation is created unless the variable is an input state variable; this case is handled
        specially by :meth:`_convert_state_variable_deriv`.

        If the direction is input, also creates an equation assigning ``original_variable`` from the converted one.
        See :meth:`convert_variable` for the context and examples.

        :param original_variable: :class:`Variable` object to be converted [old units]
        :param cf: conversion factor [new units/old units]
        :param units: Unit object for new units
        :param direction: enumeration value specifying input or output
        :param move_annotations: whether to point metadata annotations at the converted variable instead of the original
        :return: the new variable created [new units]
        """
        # Get unique name for new variable
        new_name = self.get_unique_name(original_variable.name + '_converted')

        # If original has initial_value calculate converted initial value (only needed for INPUT case)
        new_initial_value = None
        if direction == DataDirectionFlow.INPUT and original_variable.initial_value is not None:
            new_initial_value = original_variable.initial_value * float(cf)

        # Create new variable
        new_variable = self.add_variable(name=new_name, units=units, initial_value=new_initial_value)

        # Transfer cmeta id from original to new variable if the original variable has one,
        # so metadata annotations will point at the new variable.
        if original_variable._cmeta_id is not None and move_annotations:
            self.transfer_cmeta_id(original_variable, new_variable)

        # Add/replace equations defining the new variable and/or the original variable
        if direction == DataDirectionFlow.INPUT:
            # If the original variable was defined directly by an equation (original_variable = rhs) then this must be
            # removed, and the new variable defined in terms of that RHS, with a conversion factor:
            #     new_var [new units] = rhs [old units] * cf [new units/old units]
            original_equation = self._var_definition_map.get(original_variable)
            if original_equation is not None:
                new_equation = sympy.Eq(new_variable, original_equation.args[1] * cf)
                self.remove_equation(original_equation)
                self.add_equation(new_equation)

            # Add equation defining the original variable in terms of new variable:
            #     orig_var [old units] = new var [new units] / cf [new units/old units]
            # Note that state variables will still have an ODE at this point, and so will be overdefined, hence the need
            # to disable checking for duplicates.
            original_variable.initial_value = None  # This has moved to the new variable
            expression = sympy.Eq(original_variable, new_variable / cf)
            self.add_equation(expression, check_duplicates=original_variable not in self._ode_definition_map)
        else:
            # Output is the easy case: just add equation for new variable in terms of original variable
            #     new_var [new units] = orig_var [old units] * cf [new units/old units]
            new_equation = sympy.Eq(new_variable, original_variable * cf)
            self.add_equation(new_equation)

        return new_variable

    def remove_fixable_singularities(self, V, exclude=set()):
        """
        Removes removable singularities from the model equations and replaces these with a piecewise.

        The process looks for equations of any of the following forms, where U is a function of V:
        - `U / (exp(U) - 1.0)`
        - `U / (1.0 - exp(U))`
        - `(exp(U) - 1.0) / U`
        - `(1.0 - exp(U)) / U`
        It replaces these with a piecewise 1e-7 either side of U==0 drawing a stright line in the region.
        For example `(V + 5)/(exp(V + 5) - 1)` becomes
        `((fabs(-V - 5.0000000000000000) < fabs(-4.9999999000000000 / 2 - -5.0000001000000000 / 2))
        ? -0.494049243462503*V - 1.4702462167574 : ((5.0 + V) / (-1.0 + exp(5.0 + V))))`

        :param exclude: set of variables which will not be substituted in the evaluation.

        This ensures their defining equations will remain.
        """
        from ._singularity_fixes import remove_fixable_singularities
        if not isinstance(exclude, set):
            raise TypeError('exclude is expected to be a set')
        remove_fixable_singularities(self, V, exclude,
                                     exp_function=parser.SIMPLE_MATHML_TO_SYMPY_CLASSES['exp'])


class Quantity(sympy.Dummy):
    """
    Used to represent a number with a unit, inside a Sympy expression.

    Unlike sympy expressions, this number type will never be removed in simplify operations etc.

    Quantities should never be created directly, but always via :meth:`Model.create_quantity()`.

    Assumes the value is real.

    To get the actual value as a float or string, use ``float(dummy)`` or ``str(dummy)`` respectively.
    You can also use ``quantity.evalf()`` to get the value as a :class:`sympy.Float`.
    """

    # Sympy annoyingly overwrites __new__
    def __new__(cls, value, *args, **kwargs):
        # Middle argument is the symbol name, so must be a string
        if not isinstance(value, str):
            value = '{:g}'.format(value)
        return super().__new__(cls, '_' + value, real=True)

    def __init__(self, value, units):
        self._value = value
        self.units = units

    def __float__(self):
        return float(self._value)

    def _eval_evalf(self, prec):
        """This is needed to allow Sympy's ``evalf`` method to represent this value as a float."""
        return sympy.Float(self._value, prec)

    def __str__(self):
        return str(self._value)


class Variable(sympy.Dummy):
    """
    Used to represent a variable (with meta data) in a Sympy expression.

    Variables should never be created directly, but always via :meth:`Model.add_variable()`.

    For the constructor arguments, see :meth:`Model.add_variable()`.

    Assumes the value is real.
    """

    # Sympy annoyingly overwrites __new__
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, real=True)

    def __init__(self,
                 name=None,
                 units=None,
                 model=None,
                 initial_value=None,
                 public_interface=None,
                 private_interface=None,
                 order_added=None,
                 cmeta_id=None):
        self._model = model
        self.name = name
        self.units = units
        self.initial_value = None if initial_value is None else float(initial_value)

        # Interface properties, only used during parsing
        self.public_interface = public_interface
        self.private_interface = private_interface

        # Variables are either 'source' variables, or receive their value from
        # a variable that they're connected to (using BigModel connections).
        # The ``assigned_to`` property is used to indicate where this object
        # receives its value.
        self.assigned_to = None
        if not (private_interface == 'in' or public_interface == 'in'):
            self.assigned_to = self

        # Optional order added, used for sorting sometimes.
        self.order_added = order_added

        # Optional cmeta id
        self._cmeta_id = self._rdf_identity = None
        self._set_cmeta_id(cmeta_id)

        # This variable's type
        self.type = None

    def __str__(self):
        return self.name

    @property
    def model(self):
        """The :class:`Model` this variable is part of."""
        return self._model

    @property
    def rdf_identity(self):
        """The RDF identity for this variable (will be ``None`` unless the variable has a cmeta id)."""
        return self._rdf_identity

    @property
    def cmeta_id(self):
        """Provides read-only access to the cmeta id."""
        return self._cmeta_id

    def _set_cmeta_id(self, cmeta_id):
        """Sets this variable's cmeta id. Should only be called by Model, which can verify cmeta id uniqueness."""
        self._cmeta_id = cmeta_id
        if cmeta_id is None:
            self._rdf_identity = None
        else:
            self._rdf_identity = create_rdf_node('#' + self._cmeta_id)


class DataDirectionFlow(Enum):
    """Direction of data flow for converting units."""
    INPUT = 1
    OUTPUT = 2


class VariableType(Enum):
    """Classification of variables according to their role in the model's mathematics.

    ``UNKNOWN``
        not yet classified
    ``STATE``
        the dependent variable in an ODE
    ``FREE``
        the independent variable in an ODE
    ``PARAMETER``
        defined directly as a constant number (note that this does not include variables defined by an
        equation that evaluates as constant)
    ``COMPUTED``
        defined by any other equation
    """
    UNKNOWN = 0
    STATE = 1
    FREE = 2
    PARAMETER = 3
    COMPUTED = 4
