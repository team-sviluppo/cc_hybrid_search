from cat.looking_glass.stray_cat import StrayCat
from cat.mad_hatter.decorators import hook
from qdrant_client import models
from cat.log import log
from typing import Any, List, Dict
import time
from langchain.docstore.document import Document
from qdrant_client.qdrant_remote import QdrantRemote
from qdrant_client.http.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationSearchParams,
    CreateAliasOperation,
    CreateAlias,
    OptimizersConfigDiff,
)

hybrid_collection_name = "declarative_hybrid"
k = 5
threshold = 0.5


@hook(priority=99)
def before_cat_reads_message(user_message_json, cat):
    global k, threshold
    settings = cat.mad_hatter.get_plugin().load_settings()
    k = settings["number_of_hybrid_items"]
    threshold = settings["hybrid_threshold"]
    return user_message_json


@hook(priority=99)
def agent_fast_reply(fast_reply: Dict, cat: StrayCat) -> Dict:
    global hybrid_collection_name
    user_message: str = cat.working_memory.user_message_json.text
    if not user_message.startswith("@hybrid"):
        return fast_reply
    if user_message == "@hybrid init":
        delete_hybrid_collection_if_exists(cat, hybrid_collection_name)
        create_hybrid_collection_if_not_exists(cat, hybrid_collection_name)
        fast_reply["output"] = "Hybrid collection initialized."
        return fast_reply
    if user_message == "@hybrid migrate":
        points = get_declarative_points(cat)
        populate_hybrid_collection(points, cat)
        # add 5 seconds wait time to ensure data is committed
        time.sleep(5)
        fast_reply["output"] = "Hybrid collection populted."
        return fast_reply


@hook
def after_cat_bootstrap(cat: StrayCat):
    create_hybrid_collection_if_not_exists(cat, hybrid_collection_name)


def create_hybrid_collection_if_not_exists(cat, collection_name):
    client = cat.memory.vectors.vector_db
    dense_vector_name = "dense"
    sparse_vector_name = "sparse"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_vector_name: models.VectorParams(
                    size=client.get_collection(
                        "declarative"
                    ).config.params.vectors.size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={sparse_vector_name: models.SparseVectorParams()},
        )
        log.info("Hybrid collection created")


def delete_hybrid_collection_if_exists(cat, collection_name):
    client = cat.memory.vectors.vector_db
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
        log.info("Hybrid collection deleted")


@hook
def after_rabbithole_stored_documents(source, stored_points, cat):
    populate_hybrid_collection(stored_points, cat)


def get_declarative_points(cat):
    client = cat.memory.vectors.vector_db
    # Recupera tutti i punti
    all_points = []
    offset = None

    while True:
        result = client.scroll(
            collection_name="declarative",
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        points, next_offset = result
        all_points.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    return all_points


def populate_hybrid_collection(stored_points, cat):
    global hybrid_collection_name
    # Prepara i documenti per la collezione hybrid
    hybrid_points = []

    for point in stored_points:
        # Dense: usa l'embedding già calcolato dal Cat
        dense_embedding = point.vector

        # Sparse: usa il testo dal payload per generare sparse vector
        text = point.payload.get("page_content", "")
        sparse_document = models.Document(text=text, model="Qdrant/bm25")

        # Crea il nuovo punto hybrid
        hybrid_point = models.PointStruct(
            id=point.id,  # Mantieni lo stesso ID
            vector={"dense": dense_embedding, "sparse": sparse_document},
            payload=point.payload,  # Mantieni tutti i metadati
        )
        hybrid_points.append(hybrid_point)

    # Salva nella collezione hybrid
    cat.memory.vectors.vector_db.upsert(
        collection_name=hybrid_collection_name, points=hybrid_points
    )

    log.info(f"Added {len(hybrid_points)} points to hybrid collection")


def search_hybrid_collection(query, k, threshold, metadata, cat):
    global hybrid_collection_name
    client = cat.memory.vectors.vector_db
    dense_embedding = cat.embedder.embed_query(query)
    # Ricerca hybrid
    search_result = client.query_points(
        collection_name=hybrid_collection_name,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=[
            models.Prefetch(
                query=dense_embedding,
                using="dense",
            ),
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="sparse",
            ),
        ],
        with_payload=True,
        with_vectors=True,
        limit=k,
        score_threshold=threshold,
    ).points

    return search_result


def _qdrant_filter_from_dict(filter: dict) -> Filter:
    if not filter or len(filter) < 1:
        return None

    return Filter(
        should=[
            condition
            for key, value in filter.items()
            for condition in _build_condition(key, value)
        ]
    )


def _build_condition(key: str, value: Any) -> List[FieldCondition]:
    out = []

    if isinstance(value, dict):
        for _key, value in value.items():
            out.extend(_build_condition(f"{key}.{_key}", value))
    elif isinstance(value, list):
        for _value in value:
            if isinstance(_value, dict):
                out.extend(_build_condition(f"{key}[]", _value))
            else:
                out.extend(_build_condition(f"{key}", _value))
    else:
        out.append(
            FieldCondition(
                key=f"metadata.{key}",
                match=MatchValue(value=value),
            )
        )

    return out


@hook(priority=99)
def before_cat_recalls_declarative_memories(
    declarative_recall_config: dict, cat
) -> dict:
    global k, threshold
    declarative_recall_config["k"] = k
    declarative_recall_config["threshold"] = threshold
    return declarative_recall_config


@hook(priority=99)
def after_cat_recalls_memories(cat) -> None:
    global k, threshold
    metadata = {}
    memories = search_hybrid_collection(
        cat.working_memory.recall_query, k, threshold, metadata, cat
    )
    # convert Qdrant points to langchain.Document
    langchain_documents_from_points = []
    for m in memories:
        langchain_documents_from_points.append(
            (
                Document(
                    page_content=m.payload.get("page_content"),
                    metadata=m.payload.get("metadata") or {},
                ),
                m.score,
                m.vector,
                m.id,
            )
        )
    cat.working_memory.declarative_memories = langchain_documents_from_points
