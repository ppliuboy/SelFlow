import configparser

def config_dict(config_path):
    """Returns the config as dictionary,
    where the elements have intuitively correct types.
    """

    config = configparser.ConfigParser()
    config.read(config_path)

    d = dict()
    for section_key in config.sections():
        sd = dict()
        section = config[section_key]
        for key in section:
            val = section[key]
            try:
                sd[key] = int(val)
            except ValueError:
                try:
                    sd[key] = float(val)
                except ValueError:
                    try:
                        sd[key] = section.getboolean(key)
                    except ValueError:
                        sd[key] = val
        d[section_key] = sd
    return d

