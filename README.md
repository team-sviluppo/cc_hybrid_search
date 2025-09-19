# Hybrid Search

**Hybrid Search** is a plugin that enable hybrid search on qdrant vector database for declarative items.

## ✨ Key Features

- **⭐ No data loss**: The plugin create an additional collection (declarative_hybrid) that work in parallel with orginal declarative collection. Enabling / disabling plugin not cause any data loss on the orginal collection.
- **🌍 Multilingual**: Use Qdrant/bm25 for sparse emebddings, that support more languages (not only english)
- **⚙️ Configurable settings**: Set the number (k) of items and threshold to perform data retrivial

## 🚀 Quick Start

- Install plugin
- Enable plugin
- Restart CAT to create the hybrid collection
- Uplaod document to populate original declarative and new hybrid collections

## ⚡ Commands

The plugin support those commands:

```
@hybrid init - Wipe the hybrid collection
@hybrid migrate - Get all data in declarative collection and add to hybrid colletion perfroming an additionla sparse embedding
```
