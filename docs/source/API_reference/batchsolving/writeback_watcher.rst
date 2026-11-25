Writeback watcher
=================

.. currentmodule:: cubie.batchsolving.writeback_watcher

When cubie splits a batch into chunks, it copies each chunk's outputs
from the GPU through pinned staging buffers. A :class:`WritebackWatcher`
copies each chunk into the result arrays once the GPU has finished with
it, so the next chunk can run while the copy happens.

.. autoclass:: WritebackWatcher
    :members:
    :show-inheritance:

.. autoclass:: WritebackTask
    :members:
    :exclude-members: array_name, buffer, buffer_pool, data_shape, event, target_array
    :show-inheritance:
