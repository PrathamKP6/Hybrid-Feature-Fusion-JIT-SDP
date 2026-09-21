import pandas as pd

DATA_PATH = r"data/final_multilanguage_szz_buggy_384d.csv"

df = pd.read_csv(DATA_PATH, usecols=["project"])

# Remove missing values and duplicates
projects = df["project"].dropna().astype(str).str.strip()

# All distinct projects
distinct_projects = sorted(projects.unique())

print("=" * 60)
print("PROJECT DISTRIBUTION")
print("=" * 60)

print(f"Total commits           : {len(projects):,}")
print(f"Distinct projects       : {len(distinct_projects)}")

# Detect language from project prefix
def get_language(project: str) -> str:
    if project.startswith("apache/"):
        return "Java"
    elif project.startswith("cpp/"):
        return "C++"
    elif project.startswith("python/"):
        return "Python"
    else:
        return "Unknown"

project_df = pd.DataFrame({
    "project": distinct_projects
})

project_df["language"] = project_df["project"].apply(get_language)

# Count projects by language
language_counts = project_df["language"].value_counts()

print("\nProjects by language:")
print("-" * 40)

for language in ["Java", "C++", "Python", "Unknown"]:
    print(f"{language:<10}: {language_counts.get(language, 0)}")

print("\n" + "=" * 60)
print("DISTINCT PROJECT LIST")
print("=" * 60)

for language in ["Java", "C++", "Python", "Unknown"]:
    subset = project_df[project_df["language"] == language]

    if len(subset) > 0:
        print(f"\n{language} ({len(subset)} projects):")
        for project in subset["project"]:
            print(f"  - {project}")