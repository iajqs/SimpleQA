# # pytroch 1.3.1+
#
# import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertAdam
#
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained("D:\\SimpleQA\\model\\bert\\uncased")
#
# # Tokenized input
# text = "Who was Jim Henson ? Jim Henson was a puppeteer"
# text2 = "hello ! my name is chen"
# tokenized_text = tokenizer.tokenize(text)
# tokenized_text2 = tokenizer.tokenize(text2)
#
# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 6
# tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']
#
# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
#
# indexed_tokens2 = indexed_tokens2 + [0, 0, 0, 0, 0]
#
# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
# segments_ids2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# # Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens, indexed_tokens2])
# segments_tensors = torch.tensor([segments_ids, segments_ids2])
#
#
# # Load pre-trained model (weights)
# model = BertModel.from_pretrained("D:\\SimpleQA\\model\\bert\\bert-base-uncased\\")
# model.eval()
#
# # Predict hidden states features for each layer
# encoded_layers, _ = model(tokens_tensor, segments_tensors)
# # We have a hidden states for each of the 12 layers in model bert-base-uncased
# # assert len(encoded_layers) == 12
# print(len(encoded_layers), encoded_layers[0].size(), encoded_layers[11].size())
#
# # print(tokenizer.vocab)
# #
# # # print(type(tokenizer.vocab))
# # for token in tokenizer.vocab.items():
# #     print(token)
#
# print(model.embeddings.word_embeddings(torch.LongTensor([0, 1, 2]))[:, 0])