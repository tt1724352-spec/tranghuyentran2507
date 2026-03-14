import spacy
import benepar
from nltk import Tree
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Download and add Benepar model
benepar.download("benepar_en3")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def save_trees_to_html(doc, filename="syntax_trees.html"):
    html_parts = []

    # HTML header + CSS
    html_parts.append("""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Constituency & Dependency Trees</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .sentence-block { margin-bottom: 50px; }
            .container {
                display: flex;
                gap: 30px;
            }
            .tree-box {
                width: 50%;
                border: 1px solid #ccc;
                padding: 15px;
            }
            pre {
                font-size: 14px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    <h1>Constituency & Dependency Tree Visualization</h1>
    """)

    # Loop through sentences
    for i, sent in enumerate(doc.sents, 1):
        html_parts.append(f"<div class='sentence-block'>")
        html_parts.append(f"<h2>Sentence {i}: {sent.text}</h2>")
        html_parts.append("<div class='container'>")

        # Constituency tree
        html_parts.append("<div class='tree-box'><h3>Constituency Tree</h3>")
        try:
            parse_string = sent._.parse_string
            tree = Tree.fromstring(parse_string)
            html_parts.append(f"<pre>{tree.pformat()}</pre>")
        except Exception as e:
            html_parts.append(f"<p>Error: {e}</p>")
        html_parts.append("</div>")

        # Dependency tree
        html_parts.append("<div class='tree-box'><h3>Dependency Tree</h3>")
        dep_html = displacy.render(
            sent,
            style="dep",
            jupyter=False,
            options={"distance": 90}
        )
        html_parts.append(dep_html)
        html_parts.append("</div>")

        html_parts.append("</div></div>")

    # Close HTML
    html_parts.append("</body></html>")

    # Write file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"✅ Trees saved to {filename}")

if __name__ == "__main__":
    text = (
        "I enjoy learning linguistics. "
        "Why do students study syntax? "
        "Please analyze this sentence carefully. "
        "What a fascinating subject linguistics is!"
    )

    doc = nlp(text)
    save_trees_to_html(doc)