# import pdb;pdb.set_trace()
import os
import builtins
original_import = builtins.__import__
builtins.original_import = original_import
def custom_import(*args, **kw):
    module = original_import(*args, **kw)
    if (os.environ.get('IMPORT_NEW_GPNN',False)=='1' and
        "my_gpnn_inpainting"  in module.__name__
        and not getattr(module, "patch_is_performed", False)):
        # import jigsaw2 as module
#         import jigsaw_pyr_blend as module
        pass
        module.patch_is_performed = True

    return module

builtins.__import__ = custom_import