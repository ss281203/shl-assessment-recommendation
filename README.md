# SHL Assessment Recommendation Engine

## Overview
This project is an AI-based assessment recommendation system built using SHL’s product catalogue.  
It recommends the most relevant SHL assessments based on a given job requirement using semantic similarity.

## Problem Statement
Given a job description or hiring requirement, recommend suitable SHL assessments that best match the role.

## Solution Approach
- SHL assessment names are converted into semantic embeddings using a Sentence Transformer model.
- User job queries are also embedded into the same vector space.
- Cosine similarity is used to find the most relevant assessments.
- The solution is exposed via a FastAPI-based REST API.

## Tech Stack
- Python
- Pandas, NumPy
- Sentence-Transformers (all-MiniLM-L6-v2)
- FAISS
- FastAPI & Uvicorn

## Project Structure
