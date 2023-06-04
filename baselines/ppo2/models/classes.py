
_classes_dict = {}


def register(cls):
    """
    Registers a class, so that it can be obtained by a subsequent call of 'get' with the class name

    :param cls:
        class to register
    """
    name = cls.__name__
    if name in _classes_dict:
        raise ValueError('class "%s" already registered: %s' % (name, _classes_dict[name]))
    _classes_dict[name] = cls


def get(name):
    """
    Returns a previously registered class by name

    :param str name:
        class name

    :rtype: Class{PolicyNet]
    :return:
        class with given name
    """
    try:
        return _classes_dict[name]
    except KeyError:
        raise ValueError('no class with name "%s" was registered' % name)


def registered_list():
    """
    Returns a list of registered class names

    :return:
        list of registered class names
    """
    return [name for name in _classes_dict.keys()]
