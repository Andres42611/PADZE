# Releasing PADZE

This is the maintainer guide for cutting a release of **PADZE** and publishing it
to [PyPI](https://pypi.org/). Publishing is fully automated and secure: a GitHub
Release triggers a workflow that builds the distributions and uploads them to
PyPI using **Trusted Publishing (OIDC)** — no API tokens or long-lived secrets
are ever stored in this repository.

Import/distribution name: `padze` · Console command: `padze` · GitHub repo: `Andres42611/PADZE`.

---

## 1. What users get (the headline)

Once a version is on PyPI, anyone in the world can install and run PADZE with:

```bash
pip install padze
```

That single command pulls PADZE and its only runtime dependency (`numpy>=1.22`)
from PyPI and installs the `padze` console script. Users then run, for example:

```bash
padze --help
padze info --vcf trio.vcf --popmap trio.popmap
padze features --vcf trio.vcf --popmap trio.popmap --population-order A B C --out features.csv
```

Everything below is for maintainers producing that release.

---

## 2. Pre-release checklist: bump the version everywhere

PADZE keeps its version in **four** files. They must all agree before you tag a
release, and the version must be a new one that has never been published to PyPI
(PyPI rejects re-uploads of an existing version).

| File | What to change |
| --- | --- |
| `pyproject.toml` | `version = "X.Y.Z"` |
| `CITATION.cff` | `version: "X.Y.Z"` and `date-released: "YYYY-MM-DD"` |
| `codemeta.json` | `"version": "X.Y.Z"` |
| `docs/CHANGELOG.md` | Add a new `## [X.Y.Z] - YYYY-MM-DD` section describing the changes |

PADZE follows [Semantic Versioning](https://semver.org/): bump the patch for
fixes, the minor for backward-compatible features, and the major for breaking
changes. Commit the bump (and merge to the default branch) before releasing.

---

## 3. Build and check locally (dry run)

Always verify the distributions build cleanly before releasing. In a clean
virtual environment from the repository root:

```bash
pip install build twine
python -m build            # writes sdist + wheel to dist/
twine check dist/*         # both artifacts must report PASSED
```

`python -m build` produces `dist/padze-X.Y.Z.tar.gz` (sdist) and
`dist/padze-X.Y.Z-py3-none-any.whl` (wheel). `twine check` validates the package
metadata and the rendered README. Both must pass.

For an end-to-end smoke test, install the freshly built wheel into a separate,
empty virtual environment (simulating a real user) and exercise the CLI:

```bash
python -m venv /tmp/padze-userenv
/tmp/padze-userenv/bin/pip install dist/padze-X.Y.Z-py3-none-any.whl
/tmp/padze-userenv/bin/padze --help
/tmp/padze-userenv/bin/padze info --vcf examples/trio.vcf --popmap examples/trio.popmap
```

`dist/`, `build/`, and `*.egg-info/` are git-ignored; you can safely delete them
afterward (`rm -rf dist build src/padze.egg-info`).

---

## 4. One-time setup: register the Trusted Publisher on PyPI

This is done **once per project** (and once more if you also use TestPyPI). It
tells PyPI to trust releases coming from this repository's GitHub Actions
workflow, so no token ever needs to be created or stored.

1. Sign in to <https://pypi.org/>.
2. If the `padze` project does not exist yet, go to **Your projects → Publishing**
   and add a *pending* publisher (this lets the first release create the project).
   Otherwise open the existing project's **Settings → Publishing**.
3. Add a new **GitHub** Trusted Publisher with exactly these values:
   - **PyPI Project Name:** `padze`
   - **Owner:** `Andres42611`
   - **Repository name:** `PADZE`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
4. Save.

Then, in the GitHub repository, create an **Environment** named `pypi`
(**Settings → Environments → New environment**). This matches the
`environment: pypi` declared in `.github/workflows/publish.yml` and is the point
where you can optionally add required reviewers to gate publishing.

> **Dry run on TestPyPI (optional but recommended for the first release).**
> Repeat step 3 on <https://test.pypi.org/> for the same repo/workflow/environment,
> then temporarily point the publish step at TestPyPI by adding
> `with: { repository-url: https://test.pypi.org/legacy/ }` to the
> `pypa/gh-action-pypi-publish` step. Confirm the upload succeeds, then revert.

---

## 5. Release: publish automatically via a GitHub Release

With the version bumped and the Trusted Publisher registered, releasing is one
action:

1. Push the version-bump commit to the default branch.
2. On GitHub, go to **Releases → Draft a new release**.
3. Create a new tag `vX.Y.Z` (matching the version in `pyproject.toml`), targeting
   the default branch.
4. Give it a title and paste the relevant `docs/CHANGELOG.md` section as the notes.
5. Click **Publish release**.

Publishing the release triggers `.github/workflows/publish.yml`
(`on: release: types: [published]`). The `publish` job runs in the `pypi`
environment with `id-token: write` permission, builds the sdist and wheel with
`python -m build`, checks them with `twine`, and uploads them to PyPI via
`pypa/gh-action-pypi-publish` using short-lived OIDC credentials.

Watch the run under the repository's **Actions** tab. When it turns green, verify
the release is live:

```bash
pip install padze==X.Y.Z
```

and confirm it appears at <https://pypi.org/project/padze/>.

---

## 6. Troubleshooting

- **`Trusted publishing exchange failure` / OIDC error.** The publisher on PyPI
  does not match the workflow. Re-check owner, repo (`padze`), workflow filename
  (`publish.yml`), and environment (`pypi`) — all four must match exactly.
- **`File already exists` (HTTP 400).** That version was already uploaded. PyPI
  is immutable; bump to a new version and release again.
- **Workflow did not start.** It only runs on a **published** release, not on a
  draft or a bare tag push. Publish the release (or re-publish the draft).
- **`twine check` fails locally.** Usually a README rendering or metadata issue in
  `pyproject.toml`; fix and rebuild before releasing.
