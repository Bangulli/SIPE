def make_name_from_list(data):
    if isinstance(data, str):
        return data
    return "+".join(data)