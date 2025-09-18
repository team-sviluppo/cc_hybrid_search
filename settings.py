from pydantic import BaseModel
from cat.mad_hatter.decorators import plugin
from enum import Enum


# Plugin settings
class PluginSettings(BaseModel):
    number_of_hybrid_items: int = 5
    hybrid_threshold: float = 0.5


# hook to give the cat settings
@plugin
def settings_schema():
    return PluginSettings.schema()
