def format_text_for_field(first_name, last_name, animal_name, line_length=13, lines=3):
    """
    Formats the input text to fit into a field with a given number of lines and characters per line.
    """
    import textwrap
    text = f"{first_name} {last_name} {animal_name}"
    wrapped = textwrap.wrap(text, width=line_length)
    # Ensure exactly 'lines' lines (pad with empty strings if needed)
    wrapped = wrapped[:lines] + [""] * (lines - len(wrapped))
    return "\n".join(wrapped)

first_name = "SoName"
last_name = "27.05."
first_name = first_name + " " + last_name
animal_name = "AnimalName"
tag = "TBK_2025"
print(format_text_for_field(first_name, animal_name, tag))