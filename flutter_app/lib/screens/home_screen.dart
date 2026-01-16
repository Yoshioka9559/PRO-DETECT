import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:typed_data';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'webcam_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final String apiUrl = 'http://localhost:8000';
  XFile? _selectedImage;
  Uint8List? _imageBytes;
  String? _annotatedImageBase64;
  double _confidence = 0.25;
  int _totalDetections = 0;
  Map<String, int> _classCount = {};
  List<dynamic> _detectionDetails = [];
  bool _isDetecting = false;
  String _statusMessage = '✓ Ready. Please upload an image.';

  final ImagePicker _picker = ImagePicker();

  // Upload Image
  Future<void> _uploadImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      // Read image bytes for display
      final bytes = await image.readAsBytes();
      setState(() {
        _selectedImage = image;
        _imageBytes = bytes;
        _annotatedImageBase64 = null;
        _totalDetections = 0;
        _classCount = {};
        _detectionDetails = [];
        _statusMessage = '✓ Image loaded. Click Detect Products.';
      });
    }
  }

  // Capture from Webcam
  Future<void> _captureFromWebcam() async {
    // Navigate to webcam screen
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const WebcamScreen()),
    );
  }

  // Detect Products
  Future<void> _detectProducts() async {
    if (_selectedImage == null) {
      _showError('Please upload an image first');
      return;
    }

    setState(() {
      _isDetecting = true;
      _statusMessage = '🔍 Detecting products...';
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse('$apiUrl/detect'));
      
      // Read file bytes for upload
      final bytes = await _selectedImage!.readAsBytes();
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: _selectedImage!.name,
      ));
      request.fields['confidence'] = _confidence.toString();

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var jsonData = json.decode(responseData);

      if (jsonData['success']) {
        setState(() {
          _totalDetections = jsonData['total_detections'];
          _classCount = Map<String, int>.from(jsonData['class_counts']);
          _detectionDetails = jsonData['detection_details'];
          _annotatedImageBase64 = jsonData['annotated_image'];
          _statusMessage = '✓ Detection complete! Found $_totalDetections product(s)';
        });
      } else {
        _showError(jsonData['error']);
      }
    } catch (e) {
      _showError('Failed to detect: $e');
    } finally {
      setState(() {
        _isDetecting = false;
      });
    }
  }

  // Clear
  void _clear() {
    setState(() {
      _selectedImage = null;
      _imageBytes = null;
      _annotatedImageBase64 = null;
      _totalDetections = 0;
      _classCount = {};
      _detectionDetails = [];
      _statusMessage = '✓ Ready. Please upload an image.';
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
    setState(() {
      _statusMessage = '❌ Error: $message';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Header
            _buildHeader(),
            // Control Panel
            _buildControlPanel(),
            // Total Count Display
            _buildTotalCount(),
            // Image Display
            _buildImageDisplay(),
            // Results
            _buildResults(),
            // Status Bar
            _buildStatusBar(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      height: 120,
      color: Colors.white,
      child: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // GMI Logo
            Image.asset(
              'assets/images/gmi_logo.png',
              height: 90,
              errorBuilder: (context, error, stackTrace) {
                return const Icon(Icons.store, size: 60, color: Color(0xFF0A4A8E));
              },
            ),
            const SizedBox(width: 30),
            const Text(
              'ProDetect',
              style: TextStyle(
                fontSize: 32,
                fontWeight: FontWeight.bold,
                color: Color(0xFF0A4A8E),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildControlPanel() {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 20),
      child: Column(
        children: [
          // Buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildButton('📁 Upload Image', const Color(0xFF00d9ff), _uploadImage),
              const SizedBox(width: 10),
              _buildButton('📹 Webcam', const Color(0xFF9C27B0), _captureFromWebcam),
              const SizedBox(width: 10),
              _buildButton(
                '🔍 Detect Products',
                const Color(0xFF4CAF50),
                _isDetecting ? null : _detectProducts,
              ),
              const SizedBox(width: 10),
              _buildButton('🗑️ Clear', const Color(0xFFe94560), _clear),
            ],
          ),
          const SizedBox(height: 15),
          // Confidence Slider
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                'Confidence Threshold:',
                style: TextStyle(fontSize: 14, color: Color(0xFF0A4A8E)),
              ),
              const SizedBox(width: 10),
              SizedBox(
                width: 250,
                child: Slider(
                  value: _confidence,
                  min: 0.1,
                  max: 1.0,
                  divisions: 18,
                  activeColor: const Color(0xFF0A4A8E),
                  onChanged: (value) {
                    setState(() {
                      _confidence = value;
                    });
                  },
                ),
              ),
              const SizedBox(width: 10),
              Text(
                _confidence.toStringAsFixed(2),
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF0A4A8E),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildButton(String text, Color color, VoidCallback? onPressed) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
      child: Text(text, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
    );
  }

  Widget _buildTotalCount() {
    return Container(
      height: 80,
      color: const Color(0xFF0A4A8E),
      child: Center(
        child: Text(
          'TOTAL: $_totalDetections Products',
          style: const TextStyle(
            fontSize: 36,
            fontWeight: FontWeight.bold,
            color: Color(0xFFFFD700),
          ),
        ),
      ),
    );
  }

  Widget _buildImageDisplay() {
    return Container(
      color: const Color(0xFF0A4A8E),
      padding: const EdgeInsets.all(10),
      child: Container(
        constraints: const BoxConstraints(
          minHeight: 500,
          maxHeight: 800,
        ),
        decoration: BoxDecoration(
          color: const Color(0xFF084080),
          border: Border.all(color: const Color(0xFFFFD700), width: 3),
          borderRadius: BorderRadius.circular(8),
        ),
        child: _annotatedImageBase64 != null
            ? Image.memory(
                base64Decode(_annotatedImageBase64!),
                fit: BoxFit.contain,
              )
            : _imageBytes != null
                ? Image.memory(_imageBytes!, fit: BoxFit.contain)
                : const Center(
                    child: Text(
                      'No image loaded',
                      style: TextStyle(color: Colors.white54, fontSize: 18),
                    ),
                  ),
      ),
    );
  }

  Widget _buildResults() {
    return Container(
      color: const Color(0xFF0A4A8E),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '📊 Detection Results',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
          ),
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            height: 100,
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFF084080),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0xFFFFD700), width: 2),
            ),
            child: SingleChildScrollView(
              child: Text(
                _buildResultsText(),
                style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'Consolas'),
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _buildResultsText() {
    if (_totalDetections == 0) {
      return 'No products detected yet.\nUpload an image and click "Detect Products".';
    }

    StringBuffer buffer = StringBuffer();
    buffer.writeln('📦 TOTAL PRODUCTS DETECTED: $_totalDetections\n');
    buffer.writeln('📊 Product Breakdown:');
    _classCount.forEach((className, count) {
      buffer.writeln('   • $className: $count unit(s)');
    });
    return buffer.toString();
  }

  Widget _buildStatusBar() {
    return Container(
      height: 40,
      color: const Color(0xFF0A4A8E),
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          Text(
            _statusMessage,
            style: const TextStyle(fontSize: 12, color: Colors.white),
          ),
        ],
      ),
    );
  }
}
