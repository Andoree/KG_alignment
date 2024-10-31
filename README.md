# KG_alignment


## Training Data Format

### Training script parameters

`graph_data_dir` - Pre-processed graph data directory with three files: (i) adjacency_lists, (ii) rel2id, (iii) rela2id.
  * (ii) and (iii) are tsv-file with mapping between relation names and relation ids: <relation_name>\t<relation_id>.
  * Each line in (i) has the following format: <node_id>\t<edges>. Edges are separated with '|'. Each edge has format '<target_node>,<relation_id (rel)>,<subrelation_id (rela)>'

`train_data_dir`  - Training data directory. The directory contains:
  * (i) textual files with tokenizer input samples;
  * (ii) samples offset index files. For an example of reading an input sample see the code below (Training Data Format).

`val_data_dir` - Validation data directory. The directory structure is the same as for training data directory.

`tokenized_concepts_path` - tsv-file with mapping between node ids and tokenized concept names.
  * Each line's format is <node_id>\t<comma-separated (input_ids)>|||<comma-separated (input_ids)>...|||<comma-separated (input_ids)>, where '|||' is the separator for distinct concept names.



### Reading input samples

```python
with open(fpath, 'r', encoding="utf-8") as inp_file:
    inp_file.seek(offset)
    line = inp_file.readline()

    data = tuple(map(int, line.strip().split(',')))
    # Ending list positions for each data component
    inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]
    # input_ids of the tokenized sentence
    sentence_input_ids = data[4:inp_ids_end + 4]
    # Mask of zeros and ones indicating that input token is an entity token 
    token_entity_mask = data[inp_ids_end + 4:token_mask_end + 4]
    # Token-entity starting nodes (token nodes)
    edge_index_token_idx = data[token_mask_end + 4:ei_tok_idx_end + 4]
    # Token-entity ending nodes (entity node ids)
    edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]
```


`sentence_input_ids` - Comma-separated $L$ tokens' input_ids for a BERT-based model

`token_entity_mask` - Comma-separated list of $L$ zeros and ones indicating whether an input token is a part of an entity. 1 at $i$-th position means that $i$-th token in sentence_input_ids belongs to an entity.

`edge_index_token_idx` - Comma-separated index over tokens that have 1 in a token_entity_mask. This index describes starting nodes in a bipartite graph between tokens to graph nodes. The list length is $E$ which is equal to the length of edge_index_entity_idx.

`edge_index_entity_idx` - Comma-separated list of nodes that the tokens from edge_index_token_idx correspond to. This index describes ending nodes in a bipartite graph between tokens to graph nodes. The list length is $E$ which is equal to the length of edge_index_token_idx.



