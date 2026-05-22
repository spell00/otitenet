import sys

with open('app.py', 'r') as f:
    lines = f.readlines()

new_lines = []
in_tab1 = False
for i, line in enumerate(lines):
    if i == 155:
        new_lines.append("from otitenet.app.pages import admin_analytics, gradcam_gallery, inference_results, leaderboard\n")
    elif i >= 232 and i <= 979:
        if i == 232:
            new_lines.append("    with tab1:\n")
            new_lines.append("        leaderboard.render(ctx)\n")
        continue
    else:
        new_lines.append(line)

with open('app.py', 'w') as f:
    f.writelines(new_lines)
