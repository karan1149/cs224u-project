import pandas as pd
import pickle
import numpy as np
import os

def main():
	df = pd.read_csv('evaluation/concreteness/concreteness.csv')
	df.drop(columns=['Bigram', 'Conc.SD', 'Unknown', 'Total', 'Percent_known', 'SUBTLEX'])

	words = df['Word'].values.tolist()

	print('Num words', len(words))

	stoi = {word:i for i, word in enumerate(words)}

	ratings = df['Conc.M'].values

	print('Ratings shape', ratings.shape)

	print('Min rating', np.min(ratings))
	print('Max rating', np.max(ratings))
	print('Mean rating', np.mean(ratings))

	print('Median rating', np.median(ratings))

	print('20th percentile', np.percentile(ratings, 20))

	print('80th percentile', np.percentile(ratings, 80))

	abstract_words = set()
	concrete_words = set()

	concrete_threshold = np.percentile(ratings, 80)
	abstract_threshold = np.percentile(ratings, 20)

	for word in words:
		rating = ratings[stoi[word]]
		if rating >= concrete_threshold:
			concrete_words.add(word)
		if rating < abstract_threshold:
			abstract_words.add(word)

	print('Sizes of abstract and concrete sets', len(abstract_words), len(concrete_words))

	with open('evaluation/concreteness/concreteness_data.pkl', 'wb') as f:
		pickle.dump([ratings, stoi, abstract_words, concrete_words], f)

def process_similarity_datasets(basepath, newpath):
	filenames = [name for name in os.listdir(basepath) if name.endswith('.txt')]

	with open('evaluation/concreteness/concreteness_data.pkl', 'rb') as f:
		ratings, stoi, abstract_words, concrete_words = pickle.load(f)

	for filename in filenames:
		print('Processing', filename)

		curr_path = os.path.join(basepath, filename)

		triples = []
		with open(curr_path, 'r') as f:
			for line in f:
				word1, word2, label = line.strip().split()
				triples.append((word1, word2, label))

		print('Found %d triples...' % len(triples))

		save_path = os.path.join(newpath, filename)
		save_path.replace('EN-', 'C-')

		written_lines = 0
		with open(save_path, 'w') as f:
			for word1, word2, label in triples:
				if word1 in concrete_words and word2 in concrete_words:
					f.write(' '.join([word1, word2, label]) + '\n')
					written_lines += 1

		print('Wrote %d triples...' % written_lines)


if __name__=='__main__':
	main()
	process_similarity_datasets('evaluation/eval-word-vectors/data/word-sim', 'evaluation/eval-word-vectors/data/concrete-word-sim')