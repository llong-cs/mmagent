import numpy as np
import random

class VideoGraph:
    def __init__(self, max_img_embeddings=10, max_audio_embeddings=10):
        """Initialize a video graph with nodes for faces, voices and text events.
        
        Args:
            max_img_embeddings: Maximum number of image embeddings per face node
            max_audio_embeddings: Maximum number of audio embeddings per voice node
        """
        self.nodes = {}  # node_id -> node object
        self.edges = {}  # (node_id1, node_id2) -> edge weight
        self.max_img_embeddings = max_img_embeddings
        self.max_audio_embeddings = max_audio_embeddings
        self.next_node_id = 0

        # Maintain ordered text nodes
        self.text_nodes = []  # List of text node IDs in insertion order

    class Node:
        def __init__(self, node_id, node_type):
            self.id = node_id
            self.type = node_type  # 'img', 'voice' or 'text'
            self.embeddings = []
            self.metadata = {}

    def add_img_node(self, img_embedding):
        """Add a new face node with initial image embedding."""
        node = self.Node(self.next_node_id, 'img')
        node.embeddings.append(img_embedding)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node.id

    def add_voice_node(self, audio_embedding):
        """Add a new voice node with initial audio embedding."""
        node = self.Node(self.next_node_id, 'voice')
        node.embeddings.append(audio_embedding)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node.id

    def add_text_node(self, text):
        """Add a new text node with event description."""
        node = self.Node(self.next_node_id, 'text')
        node.metadata['text'] = text
        self.nodes[self.next_node_id] = node
        self.text_nodes.append(self.next_node_id)  # Add to ordered list
        self.next_node_id += 1
        return node.id

    def add_embedding(self, node_id, embeddings):
        """Add embeddings to an existing node. If total embeddings exceed max limit,
        randomly select embeddings to keep.
        
        Args:
            node_id: ID of target node
            embeddings: Single embedding or list of embeddings
            
        Returns:
            Boolean indicating success
        """
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
            
        if node.type == 'img':
            max_emb = self.max_img_embeddings
        elif node.type == 'voice':
            max_emb = self.max_audio_embeddings
        else:
            return False
            
        # Combine existing and new embeddings
        all_embeddings = node.embeddings + embeddings
        
        # If exceeding max limit, randomly select embeddings
        if len(all_embeddings) > max_emb:
            node.embeddings = random.sample(all_embeddings, max_emb)
        else:
            node.embeddings = all_embeddings
            
        return True

    def add_edge(self, node_id1, node_id2, weight=1.0):
        """Add or update bidirectional weighted edges between two nodes.
        Text-to-text connections are not allowed."""
        if (node_id1 in self.nodes and node_id2 in self.nodes and
            not (self.nodes[node_id1].type == 'text' and self.nodes[node_id2].type == 'text')):
            # Add both directions with same weight
            self.edges[(node_id1, node_id2)] = weight
            self.edges[(node_id2, node_id1)] = weight
            return True
        return False

    def update_edge_weight(self, node_id1, node_id2, weight):
        """Update weight of existing bidirectional edge."""
        if (node_id1, node_id2) in self.edges:
            # Update both directions
            self.edges[(node_id1, node_id2)] = weight
            self.edges[(node_id2, node_id1)] = weight
            return True
        return False

    def get_connected_nodes(self, node_id):
        """Get all nodes connected to given node."""
        connected = set()  # Use set to avoid duplicates due to bidirectional edges
        for (n1, n2), _ in self.edges.items():
            if n1 == node_id:
                connected.add(n2)
            elif n2 == node_id:
                connected.add(n1)
        return list(connected)

    def get_text_nodes_in_order(self):
        """Get text nodes in their insertion order."""
        return self.text_nodes.copy()

    def _cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(emb1, emb2) / (norm1 * norm2)

    def _average_similarity(self, embeddings1, embeddings2):
        """Calculate average cosine similarity between two lists of embeddings."""
        if not embeddings1 or not embeddings2:
            return 0
        
        similarities = []
        for emb1 in embeddings1:
            for emb2 in embeddings2:
                similarities.append(self._cosine_similarity(emb1, emb2))
        return np.mean(similarities)

    def search_img_nodes(self, query_embeddings, threshold=0.7):
        """Search for face nodes using image embeddings.
        
        Args:
            query_embeddings: Single embedding or list of embeddings
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        if not isinstance(query_embeddings, list):
            query_embeddings = [query_embeddings]
            
        results = []
        for node_id, node in self.nodes.items():
            if node.type == 'img':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                if similarity >= threshold:
                    results.append((node_id, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_voice_nodes(self, query_embeddings, threshold=0.7):
        """Search for voice nodes using audio embeddings.
        
        Args:
            query_embeddings: Single embedding or list of embeddings
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        if not isinstance(query_embeddings, list):
            query_embeddings = [query_embeddings]
            
        results = []
        for node_id, node in self.nodes.items():
            if node.type == 'voice':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                if similarity >= threshold:
                    results.append((node_id, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
