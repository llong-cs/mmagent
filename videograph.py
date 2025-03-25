"""
This module defines the VideoGraph class, which is used to represent the video graph.
"""
import random

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VideoGraph:
    """
    This class defines the VideoGraph class, which is used to represent the video graph.
    """
    def __init__(self, max_img_embeddings=10, max_audio_embeddings=20, img_matching_threshold=0.3, audio_matching_threshold=0.6, text_matching_threshold=0.75):
        """Initialize a video graph with nodes for faces, voices and text events.
        
        Args:
            max_img_embeddings: Maximum number of image embeddings per face node
            max_audio_embeddings: Maximum number of audio embeddings per voice node
        """
        self.nodes = {}  # node_id -> node object
        self.edges = {}  # (node_id1, node_id2) -> edge weight
        # Maintain ordered text nodes
        self.text_nodes = []  # List of text node IDs in insertion order
        
        self.max_img_embeddings = max_img_embeddings
        self.max_audio_embeddings = max_audio_embeddings
        
        self.img_matching_threshold = img_matching_threshold
        self.audio_matching_threshold = audio_matching_threshold
        self.text_matching_threshold = text_matching_threshold
        
        self.next_node_id = 0

    class Node:
        def __init__(self, node_id, node_type):
            self.id = node_id
            self.type = node_type  # 'img', 'voice', 'episodic' or 'semantic'
            self.embeddings = []
            self.metadata = {}
            
    def _average_similarity(self, embeddings1, embeddings2):
        """Calculate average cosine similarity between two lists of embeddings."""
        if not embeddings1 or not embeddings2:
            return 0
            
        # Convert lists to numpy arrays
        emb1_array = np.array(embeddings1)
        emb2_array = np.array(embeddings2)
        
        # Calculate pairwise cosine similarities between all embeddings
        similarities = cosine_similarity(emb1_array, emb2_array)
        
        # Return mean of all pairwise similarities
        return np.mean(similarities)

    def add_img_node(self, imgs):
        """Add a new face node with initial image embedding(s).
        
        Args:
            img_embedding: Single embedding or list of embeddings
        """
        node = self.Node(self.next_node_id, 'img')
        
        img_embeddings = imgs['embeddings']
        node.embeddings.extend(img_embeddings[:self.max_img_embeddings])
        
        node.metadata['contents'] = []
        node.metadata['contents'].extend(imgs['contents'])
        
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1

        print(f"Image node added with ID {node.id}")

        return node.id

    # TODO
    def add_voice_node(self, audios):
        """Add a new voice node with initial audio embedding(s).
        
        Args:
            audio_embedding: Single embedding or list of embeddings
        """
        node = self.Node(self.next_node_id, 'voice')
        
        audio_embeddings = audios['embeddings']
        node.embeddings.extend(audio_embeddings[:self.max_audio_embeddings])
        
        node.metadata['contents'] = []
        node.metadata['contents'].extend(audios['contents'])
        
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1

        print(f"Voice node added with ID {node.id}")

        return node.id

    def add_text_node(self, text, text_type='episodic'):
        """Add a new text node with episodic or semantic content.
        
        Args:
            text: Text content
            text_type: Type of text node ('episodic' or 'semantic')
        """
        if text_type not in ['episodic', 'semantic']:
            raise ValueError("text_type must be either 'episodic' or 'semantic'")

        node = self.Node(self.next_node_id, text_type)
        node.embeddings.append(text['embedding'])
        node.metadata['contents'] = [text['content']]
        
        self.nodes[self.next_node_id] = node
        self.text_nodes.append(self.next_node_id)  # Add to ordered list

        self.next_node_id += 1

        print(f"Text node of type {text_type} added with ID {node.id} and content: {text['content']}")

        return node.id

    # TODO
    def update_node(self, node_id, update_info):
        """Update an existing node.
        
        Args:
            node_id: ID of target node
            update_info: Dictionary of update information
            
        Returns:
            Boolean indicating success
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        
        node.metadata['contents'].extend(update_info['contents'])
        
        embeddings = update_info['embeddings']

        if node.type == 'img':
            max_emb = self.max_img_embeddings
        elif node.type == 'voice':
            max_emb = self.max_audio_embeddings
        else:
            raise ValueError("Node type must be either 'img' or'voice' to add embeddings")

        # Combine existing and new embeddings
        all_embeddings = node.embeddings + embeddings

        # If exceeding max limit, randomly select embeddings
        if len(all_embeddings) > max_emb:
            node.embeddings = random.sample(all_embeddings, max_emb)
        else:
            node.embeddings = all_embeddings
        
        print(f"Node {node_id} updated with {len(embeddings)} embeddings")

        return True

    def add_edge(self, node_id1, node_id2, weight=1.0):
        """Add or update bidirectional weighted edges between two nodes.
        Text-to-text connections are not allowed between same type text nodes."""
        if (node_id1 in self.nodes and node_id2 in self.nodes and not (self.nodes[node_id1].type == self.nodes[node_id2].type and self.nodes[node_id1].type in ['episodic', 'semantic'])):
            # Add both directions with same weight
            self.edges[(node_id1, node_id2)] = weight
            self.edges[(node_id2, node_id1)] = weight
            print(f"Edge added between {node_id1} and {node_id2}")
            return True
        return False

    def update_edge_weight(self, node_id1, node_id2, delta_weight):
        """Update weight of existing bidirectional edge."""
        if (node_id1, node_id2) in self.edges:
            # Update both directions
            self.edges[(node_id1, node_id2)] += delta_weight
            self.edges[(node_id2, node_id1)] += delta_weight
            # if the weight is less than or equal to 0, remove the edge
            if self.edges[(node_id1, node_id2)] <= 0:
                del self.edges[(node_id1, node_id2)]
                del self.edges[(node_id2, node_id1)]
                print(f"Edge removed between {node_id1} and {node_id2}")
            return True
        return False

    def reinforce_node(self, node_id, delta_weight=1):
        """Reinforce all edges connected to the given node.
        
        Args:
            node_id: ID of the node to reinforce
            delta_weight: Amount to increase edge weights by (default: 1)
            
        Returns:
            int: Number of edges reinforced
        """
        if node_id not in self.nodes:
            return 0
            
        reinforced_count = 0
        for (n1, n2) in list(self.edges.keys()):  # Create a list to avoid modification during iteration
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, delta_weight)
                reinforced_count += 1

        print(f"{reinforced_count} edges reinforced for node {node_id}")
                
        return reinforced_count

    def weaken_node(self, node_id, delta_weight=1):
        """Weaken all edges connected to the given node.
        
        Args:
            node_id: ID of the node to weaken
            delta_weight: Amount to decrease edge weights by (default: 1)
            
        Returns:
            int: Number of edges weakened
        """
        if node_id not in self.nodes:
            return 0
            
        weakened_count = 0
        for (n1, n2) in list(self.edges.keys()):  # Create a list to avoid modification during iteration
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, -delta_weight)  # Use negative delta_weight to decrease
                weakened_count += 1

        print(f"{weakened_count} edges weakened for node {node_id}")
                
        return weakened_count

    def get_connected_nodes(self, node_id, type=['img', 'voice', 'episodic', 'semantic']):
        """Get all nodes connected to given node."""
        connected = set()  # Use set to avoid duplicates due to bidirectional edges
        for (n1, n2), _ in self.edges.items():
            if n1 == node_id and self.nodes[n2].type in type:
                connected.add(n2)
            elif n2 == node_id and self.nodes[n1].type in type:
                connected.add(n1)
        return list(connected)

    def search_img_nodes(self, query_embeddings):
        """Search for face nodes using image embeddings.
        
        Args:
            query_embeddings: Single embedding or list of embeddings
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        if not isinstance(query_embeddings[0], list):
            query_embeddings = [query_embeddings]

        threshold = self.img_matching_threshold

        results = []
        for node_id, node in self.nodes.items():
            if node.type == 'img':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                if similarity >= threshold:
                    results.append((node_id, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_voice_nodes(self, audio_info):
        """Search for voice nodes using audio embeddings.
        
        Args:
            query_embeddings: Single embedding or list of embeddings
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        query_embeddings = audio_info["embeddings"]
        contents = audio_info["contents"]

        threshold = self.audio_matching_threshold

        results = []
        for node_id, node in self.nodes.items():
            if node.type == 'voice':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                # print(similarity)
                if similarity >= threshold:
                    results.append((node_id, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_entity_info(self, anchor_nodes, drop_threshold=0.9):
        """Get information about entities by retrieving connected episodic and semantic nodes.
        
        This function takes a list of anchor nodes and finds all connected image and voice nodes as entity nodes.
        For each entity node, it retrieves all connected episodic nodes and semantic nodes. The semantic nodes
        are filtered to remove redundant information by comparing similarity between node embeddings.

        Args:
            anchor_nodes (list): List of node IDs to use as anchor points for finding entities
            drop_threshold (float): Similarity threshold above which semantic nodes are considered redundant (default: 0.9)
            
        Returns:
            list: List of node IDs for episodic and filtered semantic nodes connected to all found entities
            
        Raises:
            ValueError: If any found entity node ID is not found or is not an image/voice node
        """
        entity_nodes = set()
        for anchor_node in anchor_nodes:
            entity_nodes.update(self.get_connected_nodes(anchor_node, type=['voice', 'img']))
            
        entity_nodes = list(entity_nodes)
        
        info_nodes = []
        
        for entity_id in entity_nodes:
            if entity_id not in self.nodes or (self.nodes[entity_id].type not in ['img', 'voice']):
                raise ValueError(f"Node {entity_id} is not an image or voice node")
            connected_episodic_nodes = self.get_connected_nodes(entity_id, type=['episodic'])
            info_nodes.extend(connected_episodic_nodes)
            connected_semantic_nodes = self.get_connected_nodes(entity_id, type=['semantic'])
            
            # Filter semantic nodes by iteratively removing nodes with high similarity
            while True:
                # Check all pairs of remaining semantic nodes
                nodes_to_remove = set()
                for i, node_id1 in enumerate(connected_semantic_nodes):
                    for node_id2 in connected_semantic_nodes[i+1:]:
                        # Calculate similarity between node embeddings
                        similarity = self._average_similarity(
                            self.nodes[node_id1].embeddings,
                            self.nodes[node_id2].embeddings
                        )
                        
                        # If similarity exceeds threshold, remove the node with lower edge weight
                        if similarity > drop_threshold:
                            edge_weight1 = self.edges.get((entity_id, node_id1), 0)
                            edge_weight2 = self.edges.get((entity_id, node_id2), 0)
                            if edge_weight1 < edge_weight2:
                                nodes_to_remove.add(node_id1)
                            else:
                                nodes_to_remove.add(node_id2)
                
                # If no nodes need to be removed, we're done
                if not nodes_to_remove:
                    break
                    
                # Remove the identified nodes
                connected_semantic_nodes = [n for n in connected_semantic_nodes if n not in nodes_to_remove]
                
            info_nodes.extend(connected_semantic_nodes)
            
        return info_nodes
    
    def search_text_nodes(self, query_embedding):
        query_embedding = [query_embedding]
        threshold = self.text_matching_threshold
        
        matched_text_nodes = []
        for node_id in self.text_nodes:
            node = self.nodes[node_id]
            similarity = self._average_similarity(query_embedding, node.embeddings)
            if similarity >= threshold:
                matched_text_nodes.append((node_id, similarity))
        
        matched_text_nodes = sorted(matched_text_nodes, key=lambda x: x[1], reverse=True)
        matched_text_nodes = [node_id for node_id, _ in matched_text_nodes]
        
        return matched_text_nodes
    
    def print_voice_nodes(self):
        for node_id, node in self.nodes.items():
            if node.type == 'voice':
                print("-"*100, f"Voice Node {node_id}", "-"*100)
                print(f"Contents: {node.metadata['contents']}")
    
    def print_img_nodes(self):
        pass
