# import pytest
# from unittest.mock import AsyncMock, patch
# from fastapi import HTTPException
# from app.services.post_search_verify_service import search_verify_student
# from app.schemas.search_verify_student_schema import VerifyResponse

# @pytest.mark.asyncio
# async def test_search_verify_student_success():
#     # Mock dependencies
#     mock_index = AsyncMock()
#     mock_index.fetch.return_value = {
#         "vectors": {"123": {"values": [0.1, 0.2, 0.3]}}
#     }

#     with patch("app.services.post_search_verify_service.get_pinecone_index", return_value=mock_index):
#         with patch("app.services.post_search_verify_service.load_image", return_value="mocked_image"):
#             with patch(
#                 "app.services.post_search_verify_service.verify",
#                 return_value={"verified": True, "distance": 0.5}
#             ):
#                 # Call the service
#                 response = await search_verify_student(
#                     student_id="123",
#                     reference_img="test.jpg",
#                     model_name="Facenet",
#                     detector_backend="ssd",
#                     distance_metric="cosine",
#                     enforce_detection=True,
#                     align=True
#                 )

#                 # Assertions
#                 assert isinstance(response, VerifyResponse)
#                 assert response.verified is True
#                 assert response.distance == 0.5

# @pytest.mark.asyncio
# async def test_search_verify_student_not_found():
#     # Mock dependencies
#     mock_index = AsyncMock()
#     mock_index.fetch.return_value = {"vectors": {}}

#     with patch("app.services.post_search_verify_service.get_pinecone_index", return_value=mock_index):
#         # Call the service and expect an HTTPException
#         with pytest.raises(HTTPException) as exc_info:
#             await search_verify_student(
#                 student_id="123",
#                 reference_img="test.jpg"
#             )

#         # Assertions
#         assert exc_info.value.status_code == 404
#         assert exc_info.value.detail == "Student ID not found."