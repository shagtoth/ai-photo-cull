from pathlib import Path


def generate_html_report(results, output_path: Path):
    rows = []
    for r in results:
        path = r["path"]
        s = r["scores"]
        flag = r["flag"]
        color = {
            "TOP": "#c8f7c5",
            "KEEP": "#ffffff",
            "REJECT": "#f7c5c5",
        }.get(flag, "#ffffff")

        rows.append(f"""
<tr style="background:{color}">
  <td><img src="{path.as_posix()}" style="max-width:200px; max-height:150px;"></td>
  <td>{path.name}</td>
  <td>{s['final']:.3f}</td>
  <td>{s['sharp_norm']:.3f}</td>
  <td>{s['face_sharp']:.3f}</td>
  <td>{s['expo']:.3f}</td>
  <td>{s['noise']:.3f}</td>
  <td>{s['blur']:.3f}</td>
  <td>{s['aesth']:.3f}</td>
  <td>{flag}</td>
</tr>
""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Photo Cull Report</title>
<style>
body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 4px; font-size: 12px; }}
th {{ background: #eee; }}
</style>
</head>
<body>
<h1>AI Photo Cull Report</h1>
<table>
<thead>
<tr>
  <th>Preview</th>
  <th>File</th>
  <th>Final</th>
  <th>Sharp</th>
  <th>Face Sharp</th>
  <th>Exposure</th>
  <th>Noise</th>
  <th>Blur</th>
  <th>Aesthetic</th>
  <th>Flag</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
