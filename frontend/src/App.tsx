import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Container,
  Typography,
  Paper,
  CircularProgress,
  TextField,
  Button,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

// Configure the backend URL here
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';

const DropzoneArea = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  border: '2px dashed #ccc',
  '&:hover': {
    border: '2px dashed #666',
  },
  backgroundColor: '#fafafa',
  minHeight: '200px',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
}));

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const uploadedFile = acceptedFiles[0];
    setFile(uploadedFile);
    handleFileUpload(uploadedFile);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    multiple: false,
  });

  const handleFileUpload = async (uploadedFile: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      console.log('Upload successful:', data);
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Query failed');
      }

      const data = await response.json();
      setResponse(data.response);
    } catch (error) {
      console.error('Error querying:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        PDF RAG System
      </Typography>

      <Box sx={{ mb: 4 }}>
        <DropzoneArea {...getRootProps()}>
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          {isDragActive ? (
            <Typography>Drop the PDF file here...</Typography>
          ) : (
            <Typography>
              Drag and drop a PDF file here, or click to select
            </Typography>
          )}
          {file && (
            <Typography variant="body2" sx={{ mt: 2, color: 'success.main' }}>
              File loaded: {file.name}
            </Typography>
          )}
          {loading && <CircularProgress sx={{ mt: 2 }} />}
        </DropzoneArea>
      </Box>

      <Box sx={{ mt: 4 }}>
        <TextField
          fullWidth
          label="Ask a question about the document"
          variant="outlined"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={!file || loading}
          sx={{ mb: 2 }}
        />
        <Button
          variant="contained"
          onClick={handleQuery}
          disabled={!file || !query.trim() || loading}
          fullWidth
        >
          Ask Question
        </Button>
      </Box>

      {response && (
        <Paper sx={{ mt: 4, p: 2 }}>
          <Typography variant="h6">Response:</Typography>
          <Typography variant="body1">{response}</Typography>
        </Paper>
      )}
    </Container>
  );
};

export default App;