"""Explicit request tag shared by scheduler and worker; no GPU imports."""
def is_dense_request(request_id):
    return request_id.rsplit(':',1)[-1].endswith('|router-dense')
