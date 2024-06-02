"""
Taken from https://github.com/adamfinkelstein/siggraph-pc-rooms/blob/main/gen-fake-data.py
Credit to Adam Finkelstein (Sep-Oct 2022)
"""
import random


def get_array_indices_matching_given_val(arr, val):
    return [i for i, arr_val in enumerate(arr) if arr_val == val]


def assign_random_reviewer_among_min_count(counts, block_list=None):
    if block_list is None:
        block_list = set()
    min_count = min(counts)
    indices = get_array_indices_matching_given_val(counts, min_count)
    indices = [i for i in indices if i not in block_list]
    if len(indices) == 0:
        print('Warning: no reviewers available for assignment')
        return -1
    index = random.choice(indices)
    counts[index] += 1  # this person's review count just went up by one
    return index


def get_reviewers(n_papers, n_people, m_reviews_per_paper, fname):
    with open(fname, 'w') as f:
        line = 'Submission ID,Withdrawn,Primary,Secondary,Second Secondary\n'
        f.write(line)
        counts = [0 for i in range(n_people)]
        for i in range(n_papers):
            r1 = assign_random_reviewer_among_min_count(counts)
            block_list = {r1}
            for j in range(m_reviews_per_paper):
                r2 = assign_random_reviewer_among_min_count(counts, block_list)
                block_list = block_list.union({r2})

                line = f'p{i},False,r{r1},r{r2},\n'
                f.write(line)
    print(f'wrote {fname} with reviewer counts: {counts}')


if __name__ == '__main__':
    get_reviewers(1000, 100, 2, r"C:\CommitteeSplitter\data\input.csv")
