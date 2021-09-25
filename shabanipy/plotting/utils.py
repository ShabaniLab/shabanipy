"""General plotting utilities."""
def format_phase(value, tick_number):
    """The value are expected in unit of π

    """
    if value == 0:
        return '0'
    elif value == 1.0:
        return 'π'
    elif value == -1.0:
        return '-π'
    else:
        return f'{value:g}π'
