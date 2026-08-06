"""
This :mod:`bigmodelmanip.parser` module contains the BigModel parser and related classes. It reads a BigModel model and
stores model information in the :class:`bigmodelmanip.model.Model` class. MathML equations are translated to Sympy. RDF
is handled by RDFLib.
"""
import itertools
import logging
import os
from collections import deque
from enum import Enum

import sympy
from lxml import etree

from .model import SYMPY_SYMBOL_DELIMITER, Model


logger = logging.getLogger(__name__)


UNIT_PREFIXES = {
    'yocto': 1e-24,
    'zepto': 1e-21,
    'atto': 1e-18,
    'femto': 1e-15,
    'pico': 1e-12,
    'nano': 1e-9,
    'micro': 1e-6,
    'milli': 1e-3,
    'centi': 1e-2,
    'deci': 1e-1,
    'deca': 1e+1,
    'deka': 1e+1,
    'hecto': 1e2,
    'kilo': 1e3,
    'mega': 1e6,
    'giga': 1e9,
    'tera': 1e12,
    'peta': 1e15,
    'exa': 1e18,
    'zetta': 1e21,
    'yotta': 1e24
}


# Work around for sympy issue dealing with relationals in equations that er passed to Piecewise
# see https://github.com/sympy/sympy/issues/24086
sympy.Eq.is_Boolean = True
sympy.Ne.is_Boolean = True
sympy.Ge.is_Boolean = True
sympy.Le.is_Boolean = True
sympy.Gt.is_Boolean = True
sympy.Lt.is_Boolean = True


class XmlNs(Enum):
    """Namespaces in BigModel documents"""
    BIGMODEL = 'http://www.bigmodel.org/bigmodel/1.0#'
    CMETA = 'http://www.bigmodel.org/metadata/1.0#'
    MATHML = 'http://www.w3.org/1998/Math/MathML'
    RDF = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'


def with_ns(ns_enum, name):
    """Returns an ElementTree-friendly name with namespace in brackets"""
    return '{%s}%s' % (ns_enum.value, name)


def _dump_node(node):
    """Pretty-print an XML node."""
    return etree.tostring(node, pretty_print=True).decode()


class _Component:
    """This hold information about a BigModel component. It's for internal-use only. Once the parser
    has created the flattened bigmodelmanip.Model instance, components are no longer used"""
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.siblings = set()
        self.encapsulated = set()

    def set_parent(self, parent_name):
        """Sets the parent of this component"""
        if self.parent:
            raise ValueError('Parent of component %s already %s. Cannot set %s: multiple parents not allowed!' %
                             (self.name, self.parent, parent_name))
        self.parent = parent_name

    def add_sibling(self, sibling_name):
        """Adds a sibling for this component"""
        assert sibling_name not in self.siblings, 'Sibling component %s already added!' % sibling_name
        self.siblings.add(sibling_name)

    def add_encapsulated(self, encapsulated_name):
        """Adds an encapsulated component to this component"""
        if encapsulated_name in self.encapsulated:
            raise ValueError('Encapsulated component %s already added!' % encapsulated_name)
        self.encapsulated.add(encapsulated_name)


class Parser(object):
    """Handles parsing of BigModel files"""

    def __init__(self, filepath):
        """Initialise an instance of Parser

        :param filepath: the full filepath to the BigModel model file
        """
        self.filepath = filepath

        # A :class:`Model` object or None
        self.model = None

        # A dictionary mapping component names to _Component objects
        self.components = {}

    def parse(self, unit_store=None):
        """
        The main method that reads the XML file and extracts the relevant parts of the BigModel model
        definition.

        :param unit_store: Optional :class:`bigmodelmanip.units.UnitStore` instance; if given the model will share the
            underlying registry so that conversions between model units and those from the provided store work.
        :return: a :class:`Model` holding BigModel model definition, reading for manipulation.
        """

        # Create lxml parser
        parser = etree.XMLParser(no_network=True)

        # Parse, get ElementTree
        tree = etree.parse(self.filepath, parser)

        # Validate BigModel syntax
        self._validate(parser, tree)

        # <model> root node - initialise the model object
        model_xml = tree.getroot()

        # Raise error if component units are defined
        units_in_comp_xpath = with_ns(XmlNs.BIGMODEL, 'component') + '/' + with_ns(XmlNs.BIGMODEL, 'units')
        units_in_components = model_xml.findall(units_in_comp_xpath)
        if len(units_in_components) != 0:
            msg = 'Defining units inside components is not supported (found in component'
            msg += 's ' if len(units_in_components) > 1 else ' '
            msg += ', '.join([e.getparent().get('name') for e in units_in_components]) + ').'
            raise ValueError(msg)

        self.model = Model(model_xml.get('name'),
                           model_xml.get(with_ns(XmlNs.CMETA, 'id')),
                           unit_store=unit_store)

        # handle the child elements of <model>
        self._add_units(model_xml)
        self._add_rdf(model_xml)

        component_variables = self._add_components(model_xml)
        self._add_relationships(model_xml)

        connected_variable_mapping = self._add_connections(model_xml)
        self._add_maths(component_variables, connected_variable_mapping)

        # Canonicalise representation
        self.transform_constants()

        return self.model

    @staticmethod
    def _get_variable_name(component_name, variable_name):
        return component_name + SYMPY_SYMBOL_DELIMITER + variable_name

    def _add_rdf(self, element):
        """
        Finds all ``<RDF>`` definitions under ``<element>`` and adds them to the model.

        :param element: the BigModel parent element to search for children RDF tags
        """
        for rdf in element.iter(with_ns(XmlNs.RDF, 'RDF')):
            self.model.add_rdf(etree.tostring(rdf, encoding=str))

    def _add_units(self, model):
        """
        <model> <units> <unit /> </units> </model>
        :param model: an etree.Element
        """
        units_elements = model.findall(with_ns(XmlNs.BIGMODEL, 'units'))

        # get list of built-in bigmodel units
        from .units import _BIGMODEL_UNITS
        units_found = set(_BIGMODEL_UNITS)

        # get all the units defined in the bigmodel model
        definitions_to_add = deque()
        for units_element in units_elements:
            units_name = units_element.get('name')
            # if it's a defined base unit, we can be add immediately to the model
            if units_element.get('base_units'):
                self.model.units.add_base_unit(units_name)
                units_found.add(units_name)
            # all other units are collected (because they may depend on further user-defined units)
            else:
                unit_elements = [t.attrib for t in units_element.getchildren()]
                definitions_to_add.append((units_name, unit_elements))

        iteration = 0
        # while we still have units to add
        while definitions_to_add:
            # get a definition from the top of the list
            unit_name, unit_elements = definitions_to_add.pop()

            # check whether this unit is defined in terms of units that we know about
            add_now = True
            for unit in unit_elements:
                # if defined in terms of units we don't know about
                if unit['units'] not in units_found:
                    # defer adding this units - add it back to the end of the list
                    definitions_to_add.appendleft((unit_name, unit_elements))
                    add_now = False
                    break

            # unit is defined in terms of known units - ok to add to model
            if add_now:
                definition = self._make_pint_unit_definition(unit_name, unit_elements)
                if self.model.units.is_defined(unit_name):
                    raise ValueError(f'Duplicate unit definition {unit_name}, unit names need to be unique!')
                self.model.units.add_unit(unit_name, definition)
                units_found.add(unit_name)
                iteration = 0
            else:
                # we did not add any units in this iteration - make note
                iteration += 1

                # exit if we have not been able to add any units in the entire list of definitions
                if iteration > len(definitions_to_add):
                    raise ValueError('Cannot create units %s. Cycles or unknown units.' % definitions_to_add)

    def _make_pint_unit_definition(self, units_name, unit_attributes):
        """
        Construct and return a ``pint.UnitDefinition``.

        :param units_name: The unit name
        :param unit_attributes: A list of dicts, where each dict contains the fields
            ``{'multiplier': float, 'units': string, 'exponent': integer, 'prefix': string/integer}``.
            Not all fields are necessary but ``units`` must match a unit in Pint registry.
        """
        full_unit_expr = []

        # For each of the <unit> elements for this unit definition
        for unit_element in unit_attributes:

            # Start from the unit name
            expr = unit_element['units']

            # See https://www.bigmodel.org/specifications/bigmodel_1.1/#sec_units 5.2.2
            # offset, prefix, exponent, and multiplier

            if 'prefix' in unit_element:
                try:
                    power = UNIT_PREFIXES[unit_element['prefix']]
                except KeyError:
                    # Assume that prefix is an integer.
                    power = '1e%s' % unit_element['prefix']
                expr = '(%s * %s)' % (expr, power)

            if 'exponent' in unit_element:
                expr = '((%s)**%s)' % (expr, unit_element['exponent'])

            if 'multiplier' in unit_element:
                expr = '(%s * %s)' % (unit_element['multiplier'], expr)

            if 'offset' in unit_element and (not unit_element['offset'].strip().isnumeric() or
                                             int(unit_element['offset']) != 0):
                raise ValueError('Offsets in units are not supported!')

            # Collect/add this particular <unit> definition
            full_unit_expr.append(expr)

        # Join together all the parts of the unit expression and return
        return '*'.join(full_unit_expr)

    def _add_components(self, model):
        """
        <model> <component> </model>
        :param model: an etree.Element

        :return: a list of (element, variable_to_symbol) tuples with maths to be added later.
        """
        component_elements = model.findall(with_ns(XmlNs.BIGMODEL, 'component'))

        component_variables = []
        # for each component defined in the model
        for element in component_elements:
            # component are only kept in parser to resolve relationships and connections
            name = element.get('name')
            if name in self.components:
                raise ValueError(f'Duplicate component name {name}, component names must be unique!')
            self.components[name] = _Component(name)

            # process the <variable> tags in this component
            variable_to_symbol = self._add_variables(element)

            # to speed up parsing, we store the maths to add to perform adding later when we know how variables connect
            component_variables.append((element, variable_to_symbol))

            # Raise error if reactions are defined
            reactions = element.findall(with_ns(XmlNs.BIGMODEL, 'reaction'))
            if reactions:
                raise ValueError(
                    'Reactions are not supported (found in component ' + name + ').')
        return component_variables

    def _add_variables(self, component_element):
        """
        <model> <component> <variable> </component> </model>
        :param component_element: an etree.Element
        """
        variable_elements = component_element.findall(with_ns(XmlNs.BIGMODEL, 'variable'))

        # we keep a {variable name: sympy symbol} lookup that we pass to the transpiler
        variable_lookup_symbol = {}

        for variable_element in variable_elements:
            attributes = dict(variable_element.attrib)

            # Rename key for cmeta_id (remove namespace from attribute)
            cmeta_id_attribute = with_ns(XmlNs.CMETA, 'id')
            if cmeta_id_attribute in attributes:
                attributes['cmeta_id'] = attributes.pop(cmeta_id_attribute)

            # mangle the name by prefixing with the component name
            attributes['name'] = Parser._get_variable_name(component_element.get('name'),
                                                           attributes['name'])

            # look up units
            attributes['units'] = self.model.units.get_unit(attributes['units'])

            # model.add_variable() returns sympy dummy created for this variable - keep it
            variable_lookup_symbol[attributes['name']] = self.model.add_variable(**attributes)

        return variable_lookup_symbol

    def _add_maths(self, component_variables, connected_variable_mapping):
        """
        Add maths for all elements stored in component_variables.

        <model> <component> <math> </component> </model>

        :param component_variables: a list of (component_element, variable_to_symbol) tuples
                                    where ``component_element`` is an ``etree.Element`
                                    and ``variable_to_symbol`` is a ``Dict[str, sympy.Dummy]``
                                        with the key the variable name and the value the variable in the model

        :param connected_variable_mapping: a ``Dict[str, sympy.Dummy]`` mapping a connected variable to its source.
        """

        for component_element, variable_to_symbol in component_variables:
            # get all <math> elements in the component
            math_elements = component_element.findall(with_ns(XmlNs.MATHML, 'math'))

            # nothing to do if we don't have any <math> elements
            if math_elements:
                # Method to create symbols
                prefix = component_element.get('name') + SYMPY_SYMBOL_DELIMITER

                def symbol_generator(identifer):
                    symbol_name = prefix + identifer
                    out = variable_to_symbol.get(symbol_name, None)
                    while str(out) in connected_variable_mapping:
                        out = connected_variable_mapping[str(out)]
                    assert out is not None, '%s not found in symbol dict' % (prefix + identifer)
                    return out

                # Set up transpiler to generate correct symbol names for connected variables in one go
                # without the need for post-processing
                transpiler = Transpiler(
                    symbol_generator=symbol_generator,
                    number_generator=lambda x, y: self.model.create_quantity(x, self.model.units.get_unit(y)),
                )

                # for each math element
                for math_element in math_elements:
                    sympy_exprs = transpiler.parse_tree(math_element)

                    # add each equation from <math> to the model
                    for expr in sympy_exprs:
                        self.model.add_equation(expr)

    def _add_relationships(self, model: etree.Element):
        group_elements = model.findall(with_ns(XmlNs.BIGMODEL, 'group'))

        # find all the <group> elements
        for group_element in group_elements:

            # find the relationship for this <group>
            relationship_ref = group_element.findall(with_ns(XmlNs.BIGMODEL, 'relationship_ref'))
            if len(relationship_ref) != 1:
                raise ValueError("Expecting exactly 1 relationship_ref tag per group, got %s!" % len(relationship_ref))
            relationship = relationship_ref[0].attrib.get('relationship')

            # we only handle 'encapsulation' relationships (i.e. ignoring 'containment')
            if relationship == 'encapsulation':
                self._handle_component_ref(group_element, None)

    def _handle_component_ref(self, parent_tag, parent_component):
        # we're going to process all the siblings at the end
        siblings = []

        # for each of the child <component_ref> elements in the parent tag
        for component_ref_element in parent_tag.findall(with_ns(XmlNs.BIGMODEL, 'component_ref')):

            # get the name of the child component
            child_component = component_ref_element.attrib.get('component')

            # add it to the sibling list
            siblings.append(child_component)

            # if we have a parent component for this child component (i.e. not top-level anonymous)
            if parent_component:
                # add the relationship in the component
                self.components[parent_component].add_encapsulated(child_component)
                self.components[child_component].set_parent(parent_component)

            # descend into this <component_ref> tag to handle any children
            self._handle_component_ref(component_ref_element, child_component)

        # if there are siblings in this non-anonymous group
        if parent_component and len(siblings) > 1:
            # register each of the siblings with each other
            for component_a, component_b in itertools.product(siblings, siblings):
                if component_a != component_b:
                    self.components[component_a].add_sibling(component_b)

    def _add_connections(self, model):
        """
        :param model: an etree.Element

        :return: dict mapping a connected variable to its source variable.
        """
        connected_variable_mapping = {}
        connection_elements = model.findall(with_ns(XmlNs.BIGMODEL, 'connection'))

        # a list to collect the (source, target) connection tuples
        connections_to_process = deque()

        # for each connection in the model
        for connection in connection_elements:
            # Should have one map_components and at least one map_variables. RELAXNGV makes use of this
            map_components = connection.find(with_ns(XmlNs.BIGMODEL, 'map_components'))
            comp_1, comp_2 = (map_components.attrib.get('component_1'),
                              map_components.attrib.get('component_2'))

            if comp_1 not in self.components:
                raise ValueError(f'Cannot connect components that do not exist: {comp_1}!')
            if comp_2 not in self.components:
                raise ValueError(f'Cannot connect components that do not exist: {comp_2}!')
            # go through all tags
            for child in connection.findall(with_ns(XmlNs.BIGMODEL, 'map_variables')):
                connections_to_process.append(
                    self._determine_connection_direction(comp_1, child.attrib.get('variable_1'),
                                                         comp_2, child.attrib.get('variable_2'))
                )

        # we add the connection to the model by first connecting
        # those variables we know are source variables

        # keep processing the list of connections until we've done them all
        unchanged_loop_count = 0
        while connections_to_process:
            # get a connection
            connection = connections_to_process.popleft()

            # if we did not successfully connect this variable (because, e.g., we don't know the
            # source of *this* source)
            source, target = connection
            # Target should not already be assigned to
            if target.assigned_to:
                raise ValueError('Target already assigned to %s before assignment to %s' %
                                 (target.assigned_to, source.assigned_to))

            if not source.assigned_to:
                # add it back to the list
                connections_to_process.append(connection)
                unchanged_loop_count += 1
            else:
                connected_variable_mapping[target.name] = source  # Store connection for later when maths is added
                # Check whether we need a unit conversion
                cf = self.model.units.get_conversion_factor(from_unit=source.units, to_unit=target.units)
                if cf == 1:
                    target.assigned_to = source.assigned_to  # Direct substitution is possible
                    if target.cmeta_id is not None:  # In case annotation is on target instead of source
                        self.model.transfer_cmeta_id(source=target, target=source)
                else:
                    cf_quant = self.model.create_quantity(cf, target.units / source.units)  # conversion factor quant
                    self.model.add_equation(sympy.Eq(target, source.assigned_to * cf_quant))  # Add connecting equation
                    target.assigned_to = target  # The assigned variable for this variable is itself
                unchanged_loop_count = 0

            assert unchanged_loop_count <= len(connections_to_process), 'Unable to add connections to the model'
        return connected_variable_mapping

    def _determine_connection_direction(self, comp_1, var_1, comp_2, var_2):
        """Takes a BigModel connection and attempts to resolve the connect by assigning the target
        variable to the assigned source variable

        Relevant lines from the BigModel specification:

            The set of all components immediately encapsulated by the current
            component is the encapsulated set.

            Other components encapsulated by the same parent make up the
            sibling set.

            The interface exposed to the parent component and components in
            the sibling set is defined by the public_interface attribute. The
            private_interface attribute defines the interface exposed to
            components in the encapsulated set. Each interface has three possible
            values: "in", "out", and "none", where "none" indicates the absence
            of an interface.
        """
        def _are_siblings(comp_a, comp_b):
            return self.components[comp_a].parent == self.components[comp_b].parent

        def _parent_of(parent_name, child_name):
            return parent_name == self.components[child_name].parent

        # get the variable information from the model about each end of the connection
        variable_1 = self.model.get_variable_by_name(self._get_variable_name(comp_1, var_1))
        variable_2 = self.model.get_variable_by_name(self._get_variable_name(comp_2, var_2))

        # if the components are siblings (either same parent or top-level)
        if _are_siblings(comp_1, comp_2):
            # they are both connected on their public_interface
            # Validation makese sure the public_interface are in/out
            # and if they are equal it would trigger an error at the connection stage (Target already assigned)
            if variable_1.public_interface == 'out':
                return variable_1, variable_2
            else:
                return variable_2, variable_1
        else:
            # determine which component is parent of the other
            if _parent_of(comp_1, comp_2):
                parent_var, child_var = variable_1, variable_2
            else:
                parent_var, child_var = variable_2, variable_1

            # parent/child components are connected using private/public interface, respectively
            if child_var.public_interface == 'in' and parent_var.private_interface == 'out':
                return parent_var, child_var
            elif child_var.public_interface == 'out' and parent_var.private_interface == 'in':
                return child_var, parent_var
        raise ValueError('Cannot determine the source & target for connection (%s, %s) - (%s, %s)' %
                         (comp_1, var_1, comp_2, var_2))

    def transform_constants(self):
        """
        Standardise handling of 'constants'.

        Once this has been called, the only variables with an initial_value attribute will be state variables,
        and the initial value will do what it implies - hold the value the state variable should take at t=0.

        Non state variables with an initial value are treated as constants. For consistent processing later on we add
        equations defining them, and remove the initial_value attribute.
        """
        state_vars = set(self.model.get_state_variables())
        for var in set(self.model.variables()):
            if var in state_vars:
                assert var.initial_value is not None, 'State variable {} has no initial_value set'.format(var)
            elif var.initial_value is not None:
                value = self.model.create_quantity(var.initial_value, var.units)
                self.model.add_equation(sympy.Eq(var, value))
                var.initial_value = None

    def _validate(self, parser, tree):
        """
        Validates the given lxml ``tree`` against the BigModel 1.0 RELAX NG schema.

        :param parser: An `lxml.etree.XMLParser`
        :param tree: An `lxml.etree.ElementTree` made with the given parser.
        """

        # Create RelaxNG object
        path = os.path.join(os.path.dirname(__file__), 'data', 'bigmodel_1_0.rng')
        rnc = etree.RelaxNG(etree.parse(path, parser))

        # Validate
        if not rnc.validate(tree):
            msg = '. '.join([str(x) for x in rnc.error_log])
            raise ValueError('Invalid or unsupported BigModel file. ' + msg)


class Transpiler(object):
    """
    Handles conversion of MathmL to Sympy exprerssions.

    :param symbol_generator: An optional method to create expressions for symbols.
        Must have signature ``f(name) -> sympy.Basic``.
    :param number_generator: An optional method to create expressions for numbers with units.
        Must have signature ``f(value, unit) -> sympy.Basic``.
    """
    def __init__(self, symbol_generator=None, number_generator=None):

        # Create simple lambdas for symbol and number generators
        if symbol_generator is None:
            symbol_generator = lambda x: sympy.Symbol(x)        # noqa: E731
        if number_generator is None:
            number_generator = lambda x, y: sympy.Float(x)      # noqa: E731

        # Store symbol and number generators
        self.symbol_generator = symbol_generator
        self.number_generator = number_generator

        # Mapping MathML tag element names (keys) to appropriate handler for SymPy output (values)
        # These tags require explicit handling because they have children or context etc.
        self.handlers = {
            'apply': self._apply_handler,
            'bvar': self._bvar_handler,
            'ci': self._ci_handler,
            'cn': self._cn_handler,
            'degree': self._degree_handler,
            'diff': self._diff_handler,
            'divide': self._divide_handler,
            'log': self._log_handler,
            'logbase': self._logbase_handler,
            'math': self.transpile,
            'minus': self._minus_handler,
            'otherwise': self._otherwise_handler,
            'piece': self._piece_handler,
            'piecewise': self._piecewise_handler,
            'power': self._power_handler,
            'root': self._root_handler
        }

        # Add tags that can be handled by simple_operator_handler
        for tag_name in SIMPLE_MATHML_TO_SYMPY_CLASSES:
            self.handlers[tag_name] = self._simple_operator_handler

    @staticmethod
    def set_mathml_handler(mathml_operator, operator_class):
        """Change how the transpiler handles a given mathml_operator.

        :param mathml_operator: The name of a MathML operator e.g. 'exp', 'true' etc.
        :param operator_class: A class that can handle the given operator e.g. ``sympy.exp``, or a function that creates
            and returns a sympy object given the operands as arguments.
        """
        SIMPLE_MATHML_TO_SYMPY_CLASSES[mathml_operator] = operator_class

    def parse_string(self, xml_string):
        """
        Reads MathML content from a string and returns equivalent SymPy expressions.
        :return: A list of SymPy expressions.
        """
        parser = etree.XMLParser(no_network=True)
        tree = etree.fromstring(xml_string, parser)
        return self.parse_tree(tree)

    def parse_tree(self, math_element):
        """Accepts a <math> element and returns equivalent SymPy expressions.

        Note: math_element must be the <math> ``Element``, not the root ``ElementTree``.

        :param math_element: <math> ``etree.Element`` object
        :return: A list of SymPy expressions.
        """
        return self.transpile(math_element)

    def transpile(self, element):
        """Convert MathML to Sympy expressions.

        Descends the given MathML element node, calling the corresponding handler for child
        elements, and returns the appropriate SymPy expression.

        :param element: an etree ``Element`` of parsed MathML
        :return: a list of SymPy expressions
        """
        # Collect the parsed expression(s) (i.e. SymPy output) into list
        sympy_expressions = []

        # For each child element of this element
        for child_element in element.iterchildren(tag='*'):
            # Call the appropriate MathML handler function for this tag
            tag_name = etree.QName(child_element.tag).localname
            if tag_name in self.handlers:
                sympy_expressions.append(self.handlers[tag_name](child_element))
            else:
                # MathML handler function not found for this tag!
                raise ValueError('No handler for element <%s>' % tag_name)

        return sympy_expressions

    # TOKEN ELEMENTS ###############################################################################

    def _ci_handler(self, node):
        """MathML:  https://www.w3.org/TR/MathML2/chapter4.html#contm.ci
        SymPy: http://docs.sympy.org/latest/modules/core.html#id17
        """
        identifier = node.text.strip()
        return self.symbol_generator(identifier)

    def _cn_handler(self, node):
        """MathML: https://www.w3.org/TR/MathML2/chapter4.html#contm.cn
        SymPy: http://docs.sympy.org/latest/modules/core.html#number
        """

        # If this number is using scientific notation
        if 'type' in node.attrib:
            if node.attrib['type'] == 'e-notation':
                # A real number may also be presented in scientific notation. Such numbers have two
                # parts (a mantissa and an exponent) separated by sep. The first part is a real
                # number, while the second part is an integer exponent indicating a power of the
                # base.. For example, 12.3<sep/>5 represents 12.3 times 10^5. The default
                # presentation of this example is 12.3e5.
                if len(node) == 1 and node[0].tag == with_ns(XmlNs.MATHML, 'sep'):
                    mantissa = node.text.strip()
                    exponent = int(node[0].tail.strip())
                    number = float('%se%d' % (mantissa, exponent))
                else:
                    raise ValueError('Expecting '
                                     '<cn type="e-notation">significand<sep/>exponent</cn>.'
                                     'Got: ' + _dump_node(node))
            else:
                raise ValueError('Unimplemented type attribute for <cn>: ' + node.attrib['type'])
        else:
            number = float(node.text.strip())

        # Get units, if given
        # TODO: We're allowing these to _not_ be set for testing only. Maybe remove this option?
        units = node.get(with_ns(XmlNs.BIGMODEL, 'units'))

        return self.number_generator(number, units)

    # BASIC CONTENT ELEMENTS #######################################################################

    def _apply_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.apply
        """
        result = self.transpile(node)
        if len(result) > 1:
            expression = result[0](*(result[1:]))
        else:
            expression = result[0]
        return expression

    def _piecewise_handler(self, node):
        """MathML: https://www.w3.org/TR/MathML2/chapter4.html#contm.piecewise
        SymPy: http://docs.sympy.org/latest/modules/functions/elementary.html#piecewise

        constructor, zero or more <piece>, zero or one <otherwise>
        """
        result = self.transpile(node)
        return sympy.Piecewise(*result)

    def _piece_handler(self, node):
        """MathML: https://www.w3.org/TR/MathML2/chapter4.html#contm.piecewise
        Returns a 2-tuple defining an expression and condition
        <piece> element contains exactly two children
        """
        result = self.transpile(node)
        if len(result) != 2:
            raise ValueError('Need exactly 2 children for <piece>')
        return result[0], result[1]

    def _otherwise_handler(self, node):
        """MathML: https://www.w3.org/TR/MathML2/chapter4.html#contm.piecewise
        Returns a 2-tuple defining an expression and condition
        """
        result = self.transpile(node)
        if len(result) != 1:
            raise ValueError('More than 1 child for <otherwise>')
        return result[0], True

    # ARITHMETIC, ALGEBRA AND LOGIC ################################################################

    def _minus_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.minus
        unary arithmetic operator OR binary arithmetic operator

        From http://docs.sympy.org/latest/tutorial/manipulation.html:
        "There is no subtraction class in SymPy. x - y is represented as x + -y, or, more
        completely, x + -1*y, i.e., Add(x, Mul(-1, y))."

        * Negation (-a) is equivalent to sympy.Mul(sympy.S.NegativeOne, a)
        * Subtraction (a - b) is equivalent to sympy.Add(a, sympy.Mul(sympy.S.NegativeOne, b))
        """
        def _wrapped_minus(left_operand, right_operand=None):
            if right_operand is None:
                # unary arithmetic operator => negation
                return -left_operand
            # otherwise, binary arithmetic operator => subtraction
            return left_operand - right_operand
        return _wrapped_minus

    def _divide_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.divide
        binary arithmetic operator
        There is no class in SymPy for division. Rather, division is represented by a power of -1.

        Equivalent to sympy.Mul(a, sympy.Pow(b, sympy.S.NegativeOne))
        """
        def _wrapped_divide(dividend, divisor):
            return dividend / divisor
        return _wrapped_divide

    def _power_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.power
        binary arithmetic operator
        equivalent to sympy.Pow(a, b)
        """
        def _wrapped_power(base, exponent):
            return base ** exponent
        return _wrapped_power

    def _root_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.root
        operator taking qualifiers

        Nasty:
        The root element is used to construct roots. The kind of root to be taken is specified by a
        degree element, which should be given as the second child of the apply element enclosing the
        root element. Thus, square roots correspond to the case where degree contains the value 2,
        cube roots correspond to 3, and so on. If no degree is present, a default value of 2 is used
        """
        def _wrapped_root(first_argument, second_argument=None):
            # if no <degree> given, it's sqrt
            if second_argument is None:
                return sympy.root(first_argument, 2)
            return sympy.root(second_argument, first_argument)
        return _wrapped_root

    def _degree_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.degree
        Meaning of <degree> depends on context! We implement it for order of <bvar> in <diff> and
        the kind of root in <root>
        """
        result = self.transpile(node)
        if len(result) != 1:
            raise ValueError('Expected single value in <degree> tag.'
                             'Got: ' + _dump_node(node))
        return result[0]

    # CALCULUS AND VECTOR CALCULUS #################################################################

    def _diff_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.diff
        operator taking qualifiers
        """
        def _wrapped_diff(x_symbol, y_symbol, evaluate=False):
            if self._is_bool(x_symbol) or self._is_bool(y_symbol):
                raise TypeError(f'Boolean not allowed in a Derivative: d{y_symbol} / d{x_symbol}')
            # if bound variable element <bvar> contains <degree>, argument x_symbol is a list,
            # otherwise, it is a symbol
            if isinstance(x_symbol, list) and len(x_symbol) == 2:
                bound_variable = x_symbol[0]
                try:
                    order = int(x_symbol[1])
                except TypeError:
                    raise TypeError(f'The degree of a derivative must be an int: d{y_symbol} / d{x_symbol}')

                deriv = sympy.Derivative(y_symbol, bound_variable, order, evaluate=evaluate)
            # Otherwise, first degree derivative
            else:
                deriv = sympy.Derivative(y_symbol, x_symbol, evaluate=evaluate)

            return deriv

        return _wrapped_diff

    def _bvar_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.bvar
        NASTY: bvar element depends on the context it is being used
        In a derivative, it indicates the variable with respect to which a function is being
        differentiated.

        The bound variable <bvar> can also specify degree. In this case, we'll have two elements
        """
        result = self.transpile(node)
        if len(result) == 1:
            # Bound variable without specifying degree
            return result[0]
        elif len(result) == 2:
            return result
        else:
            raise ValueError('Do not know how to handle <bvar> ' + _dump_node(node))

    # ELEMENTARY CLASSICAL FUNCTIONS ###############################################################

    def _log_handler(self, node):
        """https://www.w3.org/TR/MathML2/chapter4.html#contm.log
        operator taking qualifiers or a unary calculus operator
        """
        def _wrapped_log(first_element, second_element=None):
            if second_element is None:
                # if no <logbase> element is present, the base is assumed to be 10
                return sympy.log(first_element, 10)

            # Has <logbase> element, which is the first_element after <log/>
            return sympy.log(second_element, first_element)
        return _wrapped_log

    def _logbase_handler(self, node):
        """Qualifier for <log>

        The log function accepts only the logbase schema. If present, the logbase schema denotes the
        base with respect to which the logarithm is being taken. Otherwise, the log is assumed to be
        base 10. When used with log, the logbase schema is expected to contain a single child schema
        otherwise an error is generated.

        Should be the first element following log, i.e. the second child of the containing apply
        element.
        """
        return self.transpile(node)[0]

    def _get_nary_relation_callback(self, sympy_relation):
        """Wraps the Sympy binary relation to handle n-ary MathML relations

        :param sympy_relation: handle for binary Sympy relation (Eq, Le, Lt, Ge, Gt)
        :return: callback used by the apply_handler to handle n-ary relations
        """
        def _wrapper_relational(*expressions):
            # If the MathML relation is chaining more than 2 expressions
            if len(expressions) > 2:
                # Convert to multiple Sympy binary relations bundled in an 'And' boolean
                relations = []
                for first, second in zip(expressions[:-1], expressions[1:]):
                    relations.append(sympy_relation(first, second))
                return sympy.And(*relations)

            if sympy_relation in (sympy.Ge, sympy.Le, sympy.Gt, sympy.Lt) and\
                    any(map(self._is_bool, expressions)):
                raise TypeError(f'Boolean not allowed in inequality: {expressions[0]} '
                                f'{sympy_relation} {expressions[1]}')

            if sympy_relation in (sympy.Eq, sympy.Ne) and any(map(self._is_bool, expressions))\
                    and not all(map(self._is_bool, expressions)):
                if any(map(lambda e: isinstance(e, sympy.Derivative), expressions)):
                    raise TypeError(f'Boolean not allowed in inequality: {expressions[0]} '
                                    f'{sympy_relation} {expressions[1]}')
                else:
                    logger.warning(f'Boolean used in part of (in)equality equation is this intentional?: '
                                   f'{expressions[0]} {sympy_relation} {expressions[1]}')

            return sympy_relation(*expressions)
        return _wrapper_relational

    def _simple_operator_handler(self, node):
        """This function handles simple MathML <tagName> to sympy.Class operators, where no unique
        handling of tag children etc. is required.
        """
        tag_name = etree.QName(node.tag).localname

        handler = SIMPLE_MATHML_TO_SYMPY_CLASSES[tag_name]

        # Some MathML relations allow chaining but Sympy relations are binary operations
        if tag_name in MATHML_NARY_RELATIONS:
            return self._get_nary_relation_callback(handler)

        return handler

    def _is_bool(self, expr):
        """This function checks whether expr is a boolean (True/False).
        """
        return isinstance(expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse))


# These MathML tags map directly to Sympy classes and don't require any extra handling
_SIMPLE_MATHML_TO_SYMPY_CLASSES = {
    'abs': sympy.Abs,
    'and': sympy.And,
    'arccos': sympy.acos,
    'arccosh': sympy.acosh,
    'arccot': sympy.acot,
    'arccoth': sympy.acoth,
    'arccsc': sympy.acsc,
    'arccsch': sympy.acsch,
    'arcsec': sympy.asec,
    'arcsech': sympy.asech,
    'arcsin': sympy.asin,
    'arcsinh': sympy.asinh,
    'arctan': sympy.atan,
    'arctanh': sympy.atanh,
    'ceiling': sympy.ceiling,
    'cos': sympy.cos,
    'cosh': sympy.cosh,
    'cot': sympy.cot,
    'coth': sympy.coth,
    'csc': sympy.csc,
    'csch': sympy.csch,
    'eq': sympy.Eq,
    'exp': sympy.exp,
    'exponentiale': sympy.E,
    'false': sympy.false,
    'floor': sympy.floor,
    'geq': sympy.Ge,
    'gt': sympy.Gt,
    'infinity': sympy.oo,
    'leq': sympy.Le,
    'ln': sympy.ln,
    'lt': sympy.Lt,
    'max': sympy.Max,
    'min': sympy.Min,
    'neq': sympy.Ne,
    'not': sympy.Not,
    'notanumber': sympy.nan,
    'or': sympy.Or,
    'pi': sympy.pi,
    'plus': sympy.Add,
    'rem': sympy.Mod,
    'sec': sympy.sec,
    'sech': sympy.sech,
    'sin': sympy.sin,
    'sinh': sympy.sinh,
    'tan': sympy.tan,
    'tanh': sympy.tanh,
    'times': sympy.Mul,
    'true': sympy.true,
    'xor': sympy.Xor,
}

# MathML tags, to SYmpy mappings that can be changes (copy to allow original mapping to be accesible)
SIMPLE_MATHML_TO_SYMPY_CLASSES = _SIMPLE_MATHML_TO_SYMPY_CLASSES.copy()

# MathML relation elements that are n-ary operators
MATHML_NARY_RELATIONS = {'eq', 'leq', 'lt', 'geq', 'gt'}
