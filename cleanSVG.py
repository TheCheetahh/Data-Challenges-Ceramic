import os
import re
import xml.etree.ElementTree as ET
import argparse


def estimate_path_complexity(d):
    """Schätzt die Komplexität eines SVG-Pfads anhand der Anzahl an Zahlen im 'd'-Attribut."""
    if not d:
        return 0
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    return len(numbers)

def filter_most_complex_black_fill(input_path, output_path):
    """
    Behalte nur das komplexeste schwarze Objekt (fill='#000000' oder 'black') im SVG.
    Entfernt alle anderen.
    """
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(input_path)
    root = tree.getroot()
    ns = {"svg": root.tag.split('}')[0].strip('{')}

    # Alle SVG-Elemente durchlaufen
    all_elements = root.findall(".//*", ns)
    black_elements = []

    for el in all_elements:
        fill = el.attrib.get("fill", "").strip().lower()
        if fill in ("#000000", "black"):
            # Komplexität berechnen – Pfade bevorzugt, sonst 0
            complexity = 0
            if "d" in el.attrib:  # z.B. <path d="...">
                complexity = estimate_path_complexity(el.attrib["d"])
            elif el.tag.endswith("polygon") or el.tag.endswith("polyline"):
                points = el.attrib.get("points", "")
                complexity = len(re.findall(r"[-+]?\d*\.?\d+", points))
            elif el.tag.endswith(("rect", "circle", "ellipse")):
                # einfache Formen bekommen niedrige Komplexität
                complexity = 4
            black_elements.append((el, complexity))

    if not black_elements:
        print("⚠️ Keine schwarzen Flächen gefunden.")
        return

    # Komplexestes Objekt auswählen
    most_complex = max(black_elements, key=lambda x: x[1])[0]
    print(f"✅ Komplexeste schwarze Fläche gefunden ({most_complex.tag})")

    # Neues SVG erstellen, gleiche Größe/Attribute übernehmen
    new_svg = ET.Element(root.tag, root.attrib)
    new_svg.append(most_complex)

    # Ergebnis speichern
    ET.ElementTree(new_svg).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Gespeichert: {output_path}")


def get_most_complex_black_fill(raw_svg):
    """
    Takes raw SVG content as a string,
    keeps only the most complex black-filled shape,
    returns cleaned SVG as a string.
    """

    # svg parser setup stuff
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(raw_svg)
    ns = {"svg": root.tag.split('}')[0].strip('{')}

    all_elements = root.findall(".//*", ns)
    black_elements = []

    # go through all svg items in the file and search for all black shapes
    for element in all_elements:
        fill = element.attrib.get("fill", "").strip().lower()
        if fill in ("#000000", "black"):
            complexity = 0
            if "d" in element.attrib:
                complexity = estimate_path_complexity(element.attrib["d"])
            elif element.tag.endswith("polygon") or element.tag.endswith("polyline"):
                points = element.attrib.get("points", "")
                complexity = len(re.findall(r"[-+]?\d*\.?\d+", points))
            elif element.tag.endswith(("rect", "circle", "ellipse")):
                complexity = 4
            black_elements.append((element, complexity))

    if not black_elements:
        return None

    # get the most complex shape as it is likely the blob we need
    most_complex = max(black_elements, key=lambda x: x[1])[0]

    # create new svg from the blob
    new_svg = ET.Element(root.tag, root.attrib)
    new_svg.append(most_complex)

    return ET.tostring(new_svg, encoding="unicode")


def clean_all_svgs(db_handler):
    """create a svg of only the black blob for all svgs in the database"""

    # set database and get all docs
    collection = db_handler.db["svg_raw"]
    docs = collection.find({})
    counter = 0

    # get the raw content of the svg and check if it already has a cleaned svg
    for doc in docs:
        raw_content = doc.get("raw_content")
        if not raw_content or doc.get("cleaned_svg"):
            continue

        # get the black blob of the raw svg
        cleaned_svg = get_most_complex_black_fill(raw_content)

        # save the clean svg to database
        if cleaned_svg:
            counter += 1
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"cleaned_svg": cleaned_svg}}
            )
            print(f"✅ Updated {doc['filename']}")
        else:
            print(f"⚠️ No black object found in {doc['filename']}")

    return f" {counter} svgs cleaned"


# Beispielverwendung
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse eines SVG-Pfads (Krümmung etc.)")
    parser.add_argument("--input_svg", type=str, required=True, help="Pfad zur Eingabe-SVG-Datei")
    parser.add_argument("--output_dir", type=str, required=False, default="cleaned_svg/", help="Ordner für bereinigtes svg")

    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.input_svg))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_dir + base_name + ".svg"
    filter_most_complex_black_fill(args.input_svg, output_path)
