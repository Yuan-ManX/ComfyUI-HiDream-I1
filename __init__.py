from .nodes import LoadHiDreamModel, GenerateHiDreamImage, SaveHiDreamImage

NODE_CLASS_MAPPINGS = {
    "LoadHiDreamModel": LoadHiDreamModel,
    "GenerateHiDreamImage": GenerateHiDreamImage,
    "SaveHiDreamImage": SaveHiDreamImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHiDreamModel": "Load HiDream Model",
    "GenerateHiDreamImage": "Generate HiDream Image",
    "SaveHiDreamImage": "Save HiDream Image",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
