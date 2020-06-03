import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import os

from nlpaug.util import Action

#text = "I have a big headache right now and I have to go to sleep"
text = "Embarrassment is a common emotion. But as a bot I don't really feel it."
#text = 'The quick brown fox jumps over the lazy dog .'
print(text)

model_dir = '/home/rohola/codes/nlp_hub/model_dir/'

# aug = naw.WordEmbsAug(
#     model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',
#     action="substitute")
# augmented_text = aug.augment(text)
# print("Original:")
# print(text)
# print("Augmented Text:")
# print(augmented_text)

aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text: (bert-base-uncased)")
print(augmented_text)

#aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text: (distilbert-base-uncased)")
print(augmented_text)

aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text: (roberta-base)")
print(augmented_text)


aug = naw.SynonymAug(aug_src='ppdb', model_path=model_dir + 'ppdb-2.0-s-all')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)