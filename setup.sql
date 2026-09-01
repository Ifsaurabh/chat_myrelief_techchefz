-- setup.sql
-- One-time database setup for policy Agentic RAG assignment.
-- Run manually via psql: psql -U postgres -f setup.sql

CREATE DATABASE policy_rag_tc;

\c policy_rag_tc

CREATE EXTENSION vector;