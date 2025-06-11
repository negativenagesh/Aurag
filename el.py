from elasticsearch import Elasticsearch, helpers # Ensure helpers is imported

mappings = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 1536},
            "metadata": {
                "properties": {
                    "file_name": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "source_document_chunk_index": {"type": "integer"},
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "description": {"type": "text"}
                        }
                    },
                    "relationships": {
                        "type": "nested",
                        "properties": {
                            "source_entity": {"type": "keyword"},
                            "target_entity": {"type": "keyword"},
                            "relation": {"type": "keyword"},
                            "relationship_description": {"type": "text"},
                            "relationship_weight": {"type": "float"}
                        }
                    }
                }
            }
        }
    }
}

client = Elasticsearch(
    "https://my-elasticsearch-project-c44c4f.es.us-east-1.aws.elastic.cloud:443",  # Or es = Elasticsearch(['http://localhost:9200']) for local
    api_key="cWR0WFU1Y0I3YS1td2g2cWdpV186VTdnNFNVWmRmWVI5dnd3WEwwOWdMQQ==" # Or use http_auth=('user', 'password')
)


index_name = "r2rtest"  

if not client.indices.exists(index=index_name):
    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Error creating index '{index_name}': {e}")
else:
    print(f"Index '{index_name}' already exists.")
    
# For put_mapping, the body should be the mapping definition itself, not the wrapper object
# This step is generally only needed if you are updating an existing mapping.
mapping_definition = mappings.get("mappings", mappings) 
try:
    mapping_response = client.indices.put_mapping(index=index_name, body=mapping_definition)
    print(f"Put mapping response: {mapping_response}")
except Exception as e:
    print(f"Error putting mapping for index '{index_name}': {e}")


# Documents to ingest
docs = [
    {
        "_index": index_name, 
        "_source": {
            "text": "Yellowstone National Park is one of the largest national parks in the United States. It ranges from the Wyoming to Montana and Idaho, and contains an area of 2,219,791 acress across three different states. Its most famous for hosting the geyser Old Faithful and is centered on the Yellowstone Caldera, the largest super volcano on the American continent. Yellowstone is host to hundreds of species of animal, many of which are endangered or threatened. Most notably, it contains free-ranging herds of bison and elk, alongside bears, cougars and wolves. The national park receives over 4.5 million visitors annually and is a UNESCO World Heritage Site.",
            "embedding": [5.152, 9.793, 1.248, 1.579, 1.085, 5.098, 0.672, 2.388, 3.792, 9.309, 1.0, 2.0, 3.0] + [0.0] * (1536 - 13),
            "metadata": {
                "file_name": "yellowstone_facts.txt",
                "doc_id": "doc_yellowstone_001",
                "source_document_chunk_index": 0,
                "entities": [
                    {"name": "Yellowstone National Park", "type": "LOCATION", "description": "A large national park in the US."},
                    {"name": "Old Faithful", "type": "LANDMARK", "description": "A famous geyser in Yellowstone."},
                    {"name": "Bison", "type": "ANIMAL", "description": "An animal species found in Yellowstone."}
                ],
                "relationships": [
                    {"source_entity": "Yellowstone National Park", "target_entity": "Old Faithful", "relation": "CONTAINS", "relationship_description": "Yellowstone Park contains the Old Faithful geyser.", "relationship_weight": 0.9},
                    {"source_entity": "Yellowstone National Park", "target_entity": "Bison", "relation": "HABITAT_OF", "relationship_description": "Bison live in Yellowstone.", "relationship_weight": 0.8}
                ]
            }
        }
    },
    {
        "_index": index_name,
        "_source": {
            "text": "Yosemite National Park is a United States National Park, covering over 750,000 acres of land in California. A UNESCO World Heritage Site, the park is best known for its granite cliffs, waterfalls and giant sequoia trees. Yosemite hosts over four million visitors in most years, with a peak of five million visitors in 2016. The park is home to a diverse range of wildlife, including mule deer, black bears, and the endangered Sierra Nevada bighorn sheep. The park has 1,200 square miles of wilderness, and is a popular destination for rock climbers, with over 3,000 feet of vertical granite to climb. Its most famous and cliff is the El Capitan, a 3,000 feet monolith along its tallest face.",
            "embedding": [4.185, 3.596, 4.643, 9.096, 6.684, 9.183, 4.818, 8.691, 8.333, 8.11, 1.1, 2.1, 3.1] + [0.0] * (1536 - 13),
            "metadata": {
                "file_name": "yosemite_info.docx",
                "doc_id": "doc_yosemite_001",
                "source_document_chunk_index": 0,
                "entities": [
                    {"name": "Yosemite National Park", "type": "LOCATION", "description": "A national park in California."},
                    {"name": "El Capitan", "type": "LANDMARK", "description": "A famous granite monolith in Yosemite."},
                    {"name": "Giant Sequoia", "type": "PLANT", "description": "Large trees found in Yosemite."}
                ],
                "relationships": [
                    {"source_entity": "Yosemite National Park", "target_entity": "El Capitan", "relation": "FEATURES", "relationship_description": "Yosemite features El Capitan.", "relationship_weight": 0.95}
                ]
            }
        }
    },
    {
        "_index": index_name,
        "_source": {
            "text": "Rocky Mountain National Park  is one of the most popular national parks in the United States. It receives over 4.5 million visitors annually, and is known for its mountainous terrain, including Longs Peak, which is the highest peak in the park. The park is home to a variety of wildlife, including elk, mule deer, moose, and bighorn sheep. The park is also home to a variety of ecosystems, including montane, subalpine, and alpine tundra. The park is a popular destination for hiking, camping, and wildlife viewing, and is a UNESCO World Heritage Site.",
            "embedding": [4.11, 0.854, 1.583, 4.586, 4.437, 1.768, 2.545, 2.439, 7.02, 7.964, 1.2, 2.2, 3.2] + [0.0] * (1536 - 13),
            "metadata": {
                "file_name": "rocky_mountain_guide.pdf",
                "doc_id": "doc_rocky_mtn_001",
                "source_document_chunk_index": 0,
                "entities": [
                    {"name": "Rocky Mountain National Park", "type": "LOCATION", "description": "A national park known for mountains."},
                    {"name": "Longs Peak", "type": "MOUNTAIN", "description": "Highest peak in Rocky Mountain National Park."},
                    {"name": "Elk", "type": "ANIMAL", "description": "Wildlife found in the park."}
                ],
                "relationships": [
                    {"source_entity": "Rocky Mountain National Park", "target_entity": "Longs Peak", "relation": "INCLUDES", "relationship_description": "Longs Peak is within Rocky Mountain National Park.", "relationship_weight": 0.9},
                    {"source_entity": "Rocky Mountain National Park", "target_entity": "Elk", "relation": "HABITAT_OF", "relationship_description": "Elk reside in Rocky Mountain National Park.", "relationship_weight": 0.85}
                ]
            }
        }
    }
]

# Ingest documents using helpers.bulk
try:
    # The helpers.bulk function expects an iterable of actions (dictionaries).
    # Each dictionary should specify the index and the document source.
    # The 'docs' list is already formatted correctly for this.
    successes, errors = helpers.bulk(client, docs, raise_on_error=False) # No need to specify index here if it's in each doc
    print(f"Successfully indexed {successes} documents.")
    if errors:
        print(f"Encountered {len(errors)} errors during bulk indexing:")
        for i, error_info in enumerate(errors): # Renamed 'error' to 'error_info' to avoid conflict
            print(f"Error {i+1}: {error_info}")
except Exception as e:
    print(f"An error occurred during bulk ingestion: {e}")