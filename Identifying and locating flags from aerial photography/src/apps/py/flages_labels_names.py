def get_labels(data):
    names = [c['name'] for c in data]
    labels_dictionary = {c['name']: i for i, c in enumerate(data)}
    num_of_labels = len(data)
    return names, labels_dictionary, num_of_labels
