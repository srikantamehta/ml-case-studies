import faiss
import numpy as np
import pickle

class FaissIndex:
    def __init__(self, index_type='brute_force', **kwargs):
        """
        Initialize FAISS index with a selected index type and variable-length keyword arguments (kwargs).
        
        :param vectors: The input vectors (numpy array) to index.
        :param index_type: Type of FAISS index ('brute_force', 'Flat', 'IVF', 'PQ', 'HNSW', 'LSH', 'IVFSQ', 'BinaryFlat', 'BinaryIVF').
        :param kwargs: Additional parameters for index customization (e.g., 'nlist', 'm', 'bits_per_subquantizer').
        """
        # self.vector_dimension = vectors.shape[1]
        self.index_type = index_type
        self.index_params = kwargs  # Contains variable-length keyword arguments for index customization

        self.metadata = []  # Store metadata for each point

        self.index = None
    
    def _create_index(self):
        """
        Create FAISS index based on the selected index type and variable-length keyword arguments (kwargs).
        """
        if self.index_type == 'brute_force' or self.index_type == 'Flat':
            return faiss.IndexFlat(self.vector_dimension)
        
        elif self.index_type == 'IVF':
            quantizer = self.index_params.get('quantizer', faiss.IndexFlat(self.vector_dimension))
            nlist = self.index_params.get('nlist', 100)  # Number of clusters
            return faiss.IndexIVFFlat(quantizer, self.vector_dimension, nlist)
        
        elif self.index_type == 'IVFPQ':
            quantizer = self.index_params.get('quantizer', faiss.IndexFlat(self.vector_dimension))
            nlist = self.index_params.get('nlist', 100)
            m = self.index_params.get('m', 8)  # Number of subquantizers
            bits_per_subquantizer = self.index_params.get('bits_per_subquantizer', 8)
            return faiss.IndexIVFPQ(quantizer, self.vector_dimension, nlist, m, bits_per_subquantizer)
        
        elif self.index_type == 'PQ':
            m = self.index_params.get('m', 8)  # Number of subquantizers
            bits_per_subquantizer = self.index_params.get('bits_per_subquantizer', 8)
            return faiss.IndexPQ(self.vector_dimension, m, bits_per_subquantizer)
        
        elif self.index_type == 'HNSW':
            hnsw_m = self.index_params.get('hnsw_m', 32)  # Number of neighbors in HNSW
            return faiss.IndexHNSWFlat(self.vector_dimension, hnsw_m)
        
        elif self.index_type == 'LSH':
            num_bits = self.index_params.get('num_bits', 8)  # Number of bits for hashing
            return faiss.IndexLSH(self.vector_dimension, num_bits)
        
        elif self.index_type == 'IVFSQ':
            quantizer = self.index_params.get('quantizer', faiss.IndexFlat(self.vector_dimension))
            nlist = self.index_params.get('nlist', 100)
            quantization_type = self.index_params.get('quantization_type', faiss.ScalarQuantizer.QT_8bit)
            return faiss.IndexIVFScalarQuantizer(quantizer, self.vector_dimension, nlist, quantization_type)
        
        elif self.index_type == 'BinaryFlat':
            return faiss.IndexBinaryFlat(self.vector_dimension)
        
        elif self.index_type == 'BinaryIVF':
            quantizer = self.index_params.get('quantizer', faiss.IndexBinaryFlat(self.vector_dimension))
            nlist = self.index_params.get('nlist', 100)
            return faiss.IndexBinaryIVF(quantizer, self.vector_dimension, nlist)
        
        else:
            raise ValueError("Unsupported index type. Choose from 'brute_force', 'Flat', 'IVF', 'IVFPQ', 'PQ', 'HNSW', 'LSH', 'IVFSQ', 'BinaryFlat', 'BinaryIVF'.")

    def add_embeddings(self, new_vector, metadata=None):
        """
        Add a single new vector to the FAISS index.
        
        :param new_vector: The new vector to add (numpy array).
        """
        
        if self.index is None:
            self.vector_dimension = new_vector.shape[1]
            self.index = self._create_index()  # Create index on first addition

        if new_vector.shape[1] != self.vector_dimension:
            raise ValueError(f"New vector must have {self.vector_dimension} dimensions.")
        
        # Train index if necessary (for IVF, PQ, IVFSQ, etc.)
        if isinstance(self.index, (faiss.IndexIVF, faiss.IndexPQ, faiss.IndexIVFScalarQuantizer)):
            self.index.train(new_vector)
        
        self.index.add(new_vector)

        if metadata is not None:
            if isinstance(metadata, list):
                if len(metadata) != new_vector.shape[0]:
                    raise ValueError("Length of metadata list must match the number of vectors being added.")
                self.metadata.extend(metadata)
            else:
                self.metadata.append(metadata)


    def get_metadata(self, index):
        """
        Retrieve the metadata for a specific vector based on the FAISS index.
        
        :param index: The index of the vector in the FAISS index.
        :return: The metadata associated with the vector.
        """
        #print(f'in get_metadata: {self.metadata}')
        if index < len(self.metadata):
            return self.metadata[index]
        else:
            raise IndexError(f"Metadata not found for index {index}. Check if metadata was properly added.")
        
    def save(self, faiss_path: str, metadata_path: str):
        """Load FAISS index from disk.

        Args:
            path: Path to load FAISS index from.
        """
        faiss.write_index(self.index, faiss_path)
        with open(metadata_path, "wb") as file:
            pickle.dump({"metadata": self.metadata,
                         "vector_dimension": self.vector_dimension,
                         "index_params": self.index_params,
                         "index_type": self.index_type, 
                         "index_params": self.index_params,}
                        , file)

    def load(self, faiss_path: str, metadata_path: str):
        self.index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as file:
            data = pickle.load(file)
        self.metadata = data['metadata']
        self.vector_dimension = data['vector_dimension']
        self.index_params = data['index_params']
        self.index_type = data['index_type']
        self.index_params = data['index_params']
        