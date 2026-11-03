# DBSCAN Clustering: Find the Shape, Keep the Noise

> **Can a computer find two hidden crescent shapes without being told there are two?**
> This notebook lets you test that idea with DBSCAN, a clustering method that can
> find oddly shaped groups and mark unusual points as noise.

Welcome! You do not need to know machine learning already. Read a section, jump
into the notebook, change a setting, and see what happens.

## 🧭 Pick your adventure

- [The big idea](#-the-big-idea)
- [Open and run the notebook](#-open-and-run-the-notebook)
- [What you will explore](#-what-you-will-explore)
- [Meet the tools](#-meet-the-tools)
- [DBSCAN, in plain English](#-dbscan-in-plain-english)
- [Try it yourself](#-try-it-yourself)
- [Quick knowledge check](#-quick-knowledge-check)
- [A note about the saved Iris result](#-a-note-about-the-saved-iris-result)
- [What you will learn](#-what-you-will-learn)

## 💡 The big idea

**Clustering** means grouping things that seem alike, without giving the
computer the right answers first.

This notebook compares two ways to make groups:

- **K-Means** asks for the number of groups first, then tries to place every
  point into a group.
- **DBSCAN** looks for crowded areas. It can discover the number of groups by
  itself, follow curved shapes, and leave lonely points ungrouped as **noise**.

You will see both methods meet the famous **two-moons** shape. Which one do you
think will follow the curves?

## 🚀 Open and run the notebook

You need Python and Jupyter Notebook or JupyterLab. In a terminal, install the
packages used in this project:

```bash
python -m pip install jupyter scikit-learn seaborn matplotlib
```

Then, from this folder, start Jupyter:

```bash
jupyter notebook
```

Open [`dbscan_clustering.ipynb`](./dbscan_clustering.ipynb), then choose **Run
All** (or run each cell from top to bottom). The notebook creates the data,
scales it, fits the clustering methods, and draws the results.

> **Tip:** The notebook was saved with Python 3.13.9. If a package gives you an
> error, check that it is installed in the same Python environment that your
> notebook is using.

## 🔬 What you will explore

### 1. Iris flowers: clustering and noise

The notebook loads the Iris data set, which has **150 flowers** and four
measurements for each flower. It scales the measurements and runs DBSCAN with:

```python
DBSCAN(eps=0.8, min_samples=5)
```

The plot shows scaled measurements 1 and 3. DBSCAN finds groups from all the
scaled measurements, not just the two shown in the plot.

The notebook also loads the Iris flower labels, but does **not** use them to
create the groups. That is a key idea in **unsupervised learning**: the model
looks for patterns without being given the answers.

### 2. Two moons: a shape test

Next, the notebook makes **300** sample points in two curved moon shapes. A
little random noise is added so the points are not perfectly neat:

```python
make_moons(n_samples=300, noise=0.05, random_state=42)
```

The data is scaled, then clustered twice:

1. **K-Means** is asked to find 2 groups.
2. **DBSCAN** looks for dense areas with `eps=0.5` and `min_samples=5`.

The notebook's takeaway is that K-Means struggles with the curved moons, while
DBSCAN can follow their shapes. Look at both plots and decide what you notice.

## 🧰 Meet the tools

| Tool | How the notebook uses it |
|---|---|
| `sklearn.datasets.load_iris` | Loads the Iris flower measurements. |
| `sklearn.datasets.make_moons` | Creates curved sample data for the shape test. |
| `sklearn.preprocessing.StandardScaler` | Puts measurements on a similar scale before distance-based clustering. |
| `sklearn.cluster.DBSCAN` | Finds dense groups and labels noise points. |
| `sklearn.cluster.KMeans` | Makes a two-group result to compare with DBSCAN. |
| `seaborn` (`sns.scatterplot`) | Draws the points and colors them by their cluster labels. |
| `matplotlib` | Provides the plotting support used by Seaborn. |

These are the **Python packages and modules imported in the notebook**. NumPy
and other packages may be installed as dependencies of scikit-learn or Seaborn,
but the notebook does not import them directly.

## 🧠 DBSCAN, in plain English

DBSCAN has two main settings:

- **`eps`**: how close points must be to count as neighbors. Think of it as the
  reach of a small circle around each point.
- **`min_samples`**: how many nearby points are needed for an area to count as
  dense.

DBSCAN uses these settings to identify:

- **Core points:** points in a sufficiently crowded area.
- **Border points:** points near a crowded area, but not surrounded by enough
  neighbors themselves.
- **Noise points:** points that do not belong to a dense group.

Noise points get the label **`-1`**. The other labels, such as `0` and `1`, are
just group names; they do not mean "better" or "worse."

### Why scale the data first?

Distance matters to DBSCAN. If one measurement uses much larger numbers than
another, it can overpower the others. `StandardScaler` helps by centering each
measurement and putting it on a similar scale.

### When might DBSCAN be useful?

Try it when groups may have unusual shapes, when you do not know how many
groups to expect, or when spotting isolated points matters. It may need careful
settings when groups have very different densities, and its results can change
when the data scale or `eps` changes.

## 🎮 Try it yourself

Change **one thing at a time**, run the relevant cells again, and compare the
plots.

### Challenge 1: Change DBSCAN's reach

In the two-moons cell, try `eps=0.3`, then `eps=0.7`.

- Do the moons stay as two groups?
- Do more points become noise, or fewer?
- What happens if `eps` becomes too large?

### Challenge 2: Ask for more neighbors

Keep `eps=0.5`, but change `min_samples` from `5` to `10`.

- Does DBSCAN need a bigger crowd before it accepts a group?
- How does the noise on the plot change?

### Challenge 3: Test a different K-Means answer

Change `n_clusters=2` to `n_clusters=3`. K-Means will still make the number of
groups you request. Does that make the curved moon shapes easier to recognize?

### Challenge 4: Look at the labels

After running a DBSCAN cell, run:

```python
labels
```

Then count how many noise points there are:

```python
(labels == -1).sum()
```

**Extra experiment:** Try a different `random_state` in `make_moons`. Does the
same setting still give a useful result?

## ❓ Quick knowledge check

<details>
<summary><strong>1. Does DBSCAN need you to tell it the number of clusters?</strong></summary>

No. It discovers groups from dense areas. Its `eps` and `min_samples` settings
still affect what it finds.
</details>

<details>
<summary><strong>2. What does the label -1 mean?</strong></summary>

It marks a point DBSCAN treated as noise, not as part of a cluster.
</details>

<details>
<summary><strong>3. Why does the notebook scale the measurements?</strong></summary>

Because distance-based methods can be unfairly influenced by measurements with
larger number ranges. Scaling makes the measurements more comparable.
</details>

<details>
<summary><strong>4. Which method can follow the curved moons in this example?</strong></summary>

The notebook shows DBSCAN following the moon shapes better than K-Means.
K-Means is still useful, but it does not fit every shape of data.
</details>

## 🔎 A note about the saved Iris result

The notebook's written note says it found **3** Iris noise points. However, the
saved `labels` output contains **4** values equal to `-1`. Re-run the Iris
cells and use `(labels == -1).sum()` to check the result in your environment.
This is a good reminder to verify a written summary against the actual output.

## ✅ What you will learn

By the end, you will have seen how to:

- prepare measurements with `StandardScaler`;
- use DBSCAN to find groups and detect noise;
- read cluster labels, including the special noise label `-1`;
- compare a density-based method with K-Means;
- see why curved data can be a challenge for K-Means; and
- explore how DBSCAN's settings affect its results.

**Ready?** Open the notebook, run it from top to bottom, and see if your eyes
agree with the plots!
