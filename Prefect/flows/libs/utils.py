def subdivide_list(lst, chunk_size):
    # Split into batches of size chunk_size
    output = []
    for i in range(0, len(lst), chunk_size):
        output.append(lst[i : i + chunk_size])

    return output
