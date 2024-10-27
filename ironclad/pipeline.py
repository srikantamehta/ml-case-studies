from modules.extraction.embedding import Embedding
from modules.extraction.processing import Preprocessing
from modules.retrieval.index import FaissIndex
from modules.retrieval.search import FaissSearch
import os
import numpy as np
from PIL import Image

class Pipeline:

    def __init__(self, pretrained='casia-webface', device='cpu', image_size=160, index_type='brute_force', **index_params):
        """
        Initialize the Pipeline with an embedding model, preprocessing steps, and a FAISS index with customizable parameters.
        
        :param pretrained: Pretrained model ('casia-webface' or 'vggface2').
        :param device: Device to use ('cpu' or 'cuda').
        :param image_size: Size of the images.
        :param index_type: FAISS index type.
        :param index_params: Additional parameters for FAISS index customization.
        """
        # Initialize the embedding model and preprocessing
        self.embedding_model = Embedding(pretrained=pretrained, device=device)
        self.preprocessing = Preprocessing(image_size=image_size)
        
        # Initialize FAISS index with type and parameters
        self.index = FaissIndex(index_type=index_type, **index_params)

    def __encode(self, image):
        """
        Given an image path, extract the embedding vector.
        """
        raw_image = Image.open(image)
        processed_image = self.preprocessing.process(raw_image)
        embedding_vector = self.embedding_model.encode(processed_image)

        return embedding_vector

    def __precompute(self, gallery_directory):
        """
        Extract embeddings for all images in the gallery directory, collect them into a batch,
        and then store them in the FAISS index in one go.
        """
        # Initialize lists to collect embeddings and metadata
        all_embeddings = []
        all_metadata = []

        # Iterate through the gallery directory to extract embeddings
        for root, _, files in os.walk(gallery_directory):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(root, file)
                    print(image_path)
                    embedding_vector = self.__encode(image_path).reshape(1, -1)
                    all_embeddings.append(embedding_vector)
                    metadata = {'name': os.path.basename(root), 'filename': file}
                    all_metadata.append(metadata)

        # Convert the collected embeddings list to a numpy array
        all_embeddings = np.vstack(all_embeddings)

        # Add embeddings and metadata to the FAISS index in one go
        self.index.add_embeddings(all_embeddings, metadata=all_metadata)

    def __save_embeddings(self, faiss_path='ironclad/storage/catalog/faiss.index', metadata_path='ironclad/storage/catalog/metadata.pkl'):
        """
        Save the FAISS index and its metadata to disk in a binary format.
        Create the directory if it doesn't exist.
        """
        # Ensure the directory exists
        directory = os.path.dirname(faiss_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the FAISS index and metadata
        self.index.save(faiss_path, metadata_path)

    def search_gallery(self, probe_image_path, k=5, metric='euclidean', p=3, faiss_path='ironclad/storage/catalog/faiss.index', metadata_path='ironclad/storage/catalog/metadata.pkl'):
        """
        Perform a search for the k-nearest neighbors of the probe image and return their metadata.
        """
        # Load FAISS index and metadata
        self.index.load(faiss_path, metadata_path)
        # Encode the probe image and reshape 
        probe_vector = self.__encode(probe_image_path).reshape(1, -1)  
        # Initialize search with the FAISS index
        faiss_search = FaissSearch(self.index, metric=metric)
        # Perform search and return results
        _, _, metadata_results = faiss_search.search(probe_vector, k)
        return metadata_results


# if __name__ == "__main__":
#     pipeline = Pipeline()

#     # Precompute embeddings for the gallery
#     gallery_dir = "ironclad/storage/multi_image_gallery"
#     pipeline._Pipeline__precompute(gallery_dir)
    
#     # Save the computed embeddings
#     pipeline._Pipeline__save_embeddings()

#     # # Search for the k-nearest neighbors of a probe image
#     # probe_image = "ironclad/simclr_resources/probe/Will_Smith/Will_Smith_0002.jpg"
#     # results = pipeline.search_gallery(probe_image, k=5, metric='euclidean')
#     # print(results)