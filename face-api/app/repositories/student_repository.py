
class StudentRepository:
    
    
    def __init__(self, pinecone_index):
        self.index = pinecone_index
        
        
    def get_student_embedding(self, student_id: str):
        """
        Fetch the stored embedding for a given student ID from the database.
        
        Args:
            student_id (str): The student ID to search for.
        
        Returns:
            dict: The stored embedding for the student.
        """
        fetch_response = self.index.fetch(ids=[student_id])
        
        if not fetch_response.vectors or student_id not in fetch_response.vectors:
            return None
        
        return fetch_response.vectors[student_id]["values"]
    
    
    def student_exists(self, student_id: str):
        """
        Check if a student ID exists in the database.

        Args:
            student_id (str): The student ID to check.
        """
        
        fetch_response = self.index.fetch(ids=[student_id])
        
        return fetch_response.vectors and student_id in fetch_response.vectors
    
    
    def save_student_embedding(self, student_id: str, embedding, name: str):
        """
        Save a student's embedding to the database
        
        Args:
            student_id (str): The student ID to save.
            embedding: The embedding to save.
            name (str): The name of the student
        """
        
        self.index.upsert(
            
            vectors=[
                {
                    "id": student_id,
                    "values": embedding, 
                    "metadata":{
                        "student_id": student_id,
                        "name": name
                    }
                }
                
            ]
        )
        
        
    def update_student_embedding(self, student_id: str, embedding):
        """
        Update a student's embedding in the database.

        Args:
            student_id (str): The student ID to update.
            embedding (_type_): The new embedding to save.
        """
        
        self.index.update(
            
            id=student_id,
            values = embedding
        
        )
        