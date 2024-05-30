import os
import yaml


def load_yaml_config(path):
    """
    Load the configuration from a YAML file.

    If the config file doesn't exist in the user's directory, it copies the default from the project directory.

    Returns:
        dict: The loaded YAML configuration.
    """

    # Load YAML config file using the new path
    with open(path, 'r') as file:
        yaml_config = yaml.safe_load(file)

    # env vs yaml, yaml first if not empty
    for key, value in yaml_config.items():
        if os.environ.get(key) and not value:
            yaml_config[key] = os.environ.get(key)
        # if yaml has some configs that env does not, write to env
        if not os.environ.get(key) and value:
            os.environ[key] = str(value)

    return yaml_config


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(key)


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../", 'config.yaml')
config = AttrDict(load_yaml_config(DEFAULT_CONFIG_PATH))