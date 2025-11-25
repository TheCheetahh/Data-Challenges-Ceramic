def format_svg_for_display(cleaned_svg):
    """
    Wrap the SVG in a bordered white box for display on the web page.

    :param cleaned_svg: cleaned SVG
    """
    return f"""
    <div style="
        border: 2px solid black;
        background-color: white;
        padding: 10px;
        width: 500px;
        height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        {cleaned_svg}
    </div>
    """