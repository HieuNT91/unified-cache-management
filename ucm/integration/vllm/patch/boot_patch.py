"""Entry point used by uc-manager's installed ucm_patch.pth in child processes."""
def install_hook():
    from .apply_patch import install_import_hook
    install_import_hook()
