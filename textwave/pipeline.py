import os
from modules.extraction.embedding import Embedding
from modules.extraction.preprocessing import DocumentProcessing
from modules.retrieval.indexing import FaissIndex

class Pipeline:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the Pipeline with the specified embedding model, document processor, 
        and indexing method.

        :param model_name: The name of the embedding model to use. Defaults to 'all-MiniLM-L6-v2'.
        :type model_name: str
        """

        self.embedding_model = Embedding(model_name)
        self.doc_processor = DocumentProcessing()
        self.index = FaissIndex(index_type='brute_force')

    def preprocess_corpus(self, corpus_directory, chunking_strategy, fixed_length=None, overlap_size=2):
        """
        Preprocesses the text corpus in the specified directory by chunking, embedding, 
        and indexing each document.

        :param corpus_directory: Path to the directory containing text documents.
        :type corpus_directory: str
        :param chunking_strategy: The strategy for chunking documents ('sentence' or 'fixed-length').
        :type chunking_strategy: str
        :param fixed_length: Length of each chunk for fixed-length chunking. Required if 
                             chunking_strategy is 'fixed-length'.
        :type fixed_length: int, optional
        :param overlap_size: Number of overlapping sentences/words between chunks. Defaults to 2.
        :type overlap_size: int
        :return: A list of tuples, where each tuple contains a text chunk and its corresponding embedding.
        :rtype: list of tuples
        """
        processed_chunks = []
        for filename in os.listdir(corpus_directory):
            file_path = os.path.join(corpus_directory, filename)
            if os.path.isfile(file_path):
                if chunking_strategy == 'sentence':
                    chunks = self.doc_processor.sentence_chunking(file_path, overlap_size)
                elif chunking_strategy == 'fixed-length' and fixed_length:
                    chunks = self.doc_processor.fixed_length_chunking(file_path, fixed_length, overlap_size)
                else:
                    raise ValueError("Invalid chunking strategy or missing fixed_length parameter")
                
                for chunk in chunks:
                    chunk_embedding = self.embedding_model.encode(chunk)
                    self.index.add_embeddings(chunk_embedding.reshape(1,-1), metadata=chunk)
                    processed_chunks.append((chunk, chunk_embedding))
        return processed_chunks
    
if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline()
    pipeline.preprocess_corpus("textwave/storage/corpus", chunking_strategy='sentence', overlap_size=2)
    pipeline.preprocess_corpus("textwave/storage/corpus", chunking_strategy='fixed-length', fixed_length=50, overlap_size=3)