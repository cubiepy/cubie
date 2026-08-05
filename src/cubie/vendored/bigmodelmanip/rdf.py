"""Module for working with RDF data."""
import rdflib


def create_rdf_node(node_content):
    """Creates and returns an RDF node.

    :param node_content: the content for the node.

    The ``node_content``, if given, must either be a :class:`rdflib.term.Node` instance,
    a tuple ``(namespace_uri, local_name)``, or a string, in which case it is interpreted
    as either a URI ref if it starts with # otherwise a literal RDF node.

    ``node_content`` may also be ``None`` to return ``None``, allowing easy handling of wildcard options to queries.
    """
    if node_content is None or isinstance(node_content, rdflib.term.Node):
        return node_content
    elif isinstance(node_content, tuple):
        uri, local_name = node_content
        # Ensure namespace prefix can be appended to
        if not uri.endswith('#') and not uri.endswith('/'):
            uri += '#'
        return rdflib.Namespace(uri)[local_name]
    elif isinstance(node_content, str) and node_content.startswith('#'):
        return rdflib.URIRef(node_content)
    else:
        return rdflib.Literal(node_content)
