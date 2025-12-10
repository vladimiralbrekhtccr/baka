## How to init
```bash
mkdir baka
cd baka
uv venv --python 3.12 .venv
uva # alias uva='source .venv/bin/activate'
code .gitgnore
code readme.nd
uv pip install mkdocs
uv pip install mkdocs-material
mkdocs new . 
mkdocs serve -a localhost:7775
```

## git day-to-day that you will need
```bash
git add . 
git commit -m "baka"
git remote add origin git@github.com:vladimiralbrekhtccr/baka.git
git branch -M main
git push -u origin main
```

### deploy loop
Change something on main branch inside ./docs/*.md files and it use gh-deploy to deploy on git. Make sure to git push first.

```bash
`mkdocs gh-deploy`# it will create branch gh-deploy and you just need to go to settings -> pages and select branch
```
