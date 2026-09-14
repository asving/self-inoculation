"""Build blog posts for asving.github.io from markdown, in the site's existing template.
usage: python site_build.py <md> <outdir> --title T [--subtitle S] [--date D] [--byline B] [--noindex] [--figsrc DIR]
[figure: path. caption] placeholders become <figure>; the image is copied into <outdir>/figs/.
"""
import argparse, os, re, shutil, html
import markdown
ap = argparse.ArgumentParser()
ap.add_argument('md'); ap.add_argument('outdir'); ap.add_argument('--title', required=True); ap.add_argument('--subtitle', default='')
ap.add_argument('--date', default=''); ap.add_argument('--byline', default=''); ap.add_argument('--noindex', action='store_true')
ap.add_argument('--figsrc', default='.', help='directory that figure paths in the md are relative to'); ap.add_argument('--strip_h1', action='store_true')
a = ap.parse_args()
src = open(a.md).read()
if a.strip_h1: src = re.sub(r'^# .*\n', '', src, count=1)
os.makedirs(os.path.join(a.outdir, 'figs'), exist_ok=True)
def fig(m):
    body = m.group(1).strip()
    pm = re.search(r'((?:figs_h|figs|sim_h|sim)/[\w\-.]+\.png)', body)
    if not pm: return m.group(0)
    path = pm.group(1); srcp = os.path.join(a.figsrc, path)
    shutil.copy(srcp, os.path.join(a.outdir, 'figs', os.path.basename(path)))
    cap = body.replace(path, '').strip(' .,:()').strip()
    cap = re.sub(r'^\.\s*', '', cap)
    cap_html = markdown.markdown(cap).replace('<p>', '').replace('</p>', '')
    return f'\n<figure><img src="figs/{os.path.basename(path)}" alt="" style="width:100%;max-width:900px;"><figcaption><em>{cap_html}</em></figcaption></figure>\n'
src = re.sub(r'^\[figure:(.*?)\]\s*$', fig, src, flags=re.M | re.S)
def mdfig(m):
    url, cap = m.group(1), m.group(2).strip()
    cap_html = markdown.markdown(cap).replace('<p>', '').replace('</p>', '')
    return f'\n<figure><img src="{url}" alt="" style="width:100%;max-width:900px;"><figcaption><em>{cap_html}</em></figcaption></figure>\n'
src = re.sub(r'^!\[[^\]]*\]\((\S+)\)\s*\n\s*\n\*(.+?)\*\s*$', mdfig, src, flags=re.M)
body = markdown.markdown(src, extensions=['extra', 'sane_lists'])
robots = '    <meta name="robots" content="noindex">\n' if a.noindex else ''
sub = f'            <p style="font-size: 1.1rem; margin-top: -0.5rem;"><em>{html.escape(a.subtitle)}</em></p>\n' if a.subtitle else ''
date = f'            <p class="post-date">{html.escape(a.date)}</p>\n' if a.date else ''
byline = f'            <p><em>{a.byline}</em></p>\n' if a.byline else ''
page = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(a.title)} - Asvin G</title>
{robots}    <link rel="stylesheet" href="../../css/style.css">
    <style>figure {{ margin: 1.5rem 0; }} figcaption {{ font-size: 0.9rem; color: #555; margin-top: 0.4rem; }} blockquote {{ border-left: 3px solid #ccc; margin: 1rem 0; padding: 0.2rem 1rem; }}</style>
</head>
<body>
    <header>
        <img src="../../images/banner.jpg" alt="Grothendieck-Riemann-Roch" class="banner">
        <div class="container">
            <div class="site-title">
                <h1><a href="../../index.html">Asvin G</a></h1>
                <p class="tagline">Wir müssen wissen, wir werden wissen</p>
            </div>
            <nav>
                <ul>
                    <li><a href="../../index.html">About</a></li>
                    <li><a href="../../blog.html">Blog</a></li>
                    <li><a href="../../claude/index.html">Claude</a></li>
                    <li><a href="../../book-reviews.html">Book Reviews</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <main class="container-narrow">
        <article class="blog-post">
            <h1>{html.escape(a.title)}</h1>
{sub}{date}{byline}
{body}

        </article>

        <p style="margin-top: 1rem;"><a href="../../blog.html">&larr; Back to all posts</a></p>
    </main>

    <footer>
        <div class="container">
            <p>&copy; Asvin G</p>
        </div>
    </footer>
</body>
</html>
'''
open(os.path.join(a.outdir, 'index.html'), 'w').write(page)
print(f"wrote {a.outdir}/index.html ({len(body.split())} words in body; figures: {body.count('<figure>')})")
