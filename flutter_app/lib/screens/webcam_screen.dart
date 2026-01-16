import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;

class WebcamScreen extends StatefulWidget {
  const WebcamScreen({Key? key}) : super(key: key);

  @override
  State<WebcamScreen> createState() => _WebcamScreenState();
}

class _WebcamScreenState extends State<WebcamScreen> {
  final String apiUrl = 'http://localhost:8000';
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  bool _isDetecting = false;
  double _confidence = 0.25;
  int _totalDetections = 0;
  Map<String, int> _classCount = {};
  String? _annotatedImageBase64;
  String _statusMessage = 'Initializing camera...';
  bool _isFlipped = true;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        setState(() {
          _statusMessage = 'No camera found';
        });
        return;
      }

      _cameraController = CameraController(
        _cameras![0],
        ResolutionPreset.high,
      );

      await _cameraController!.initialize();
      
      if (!mounted) return;
      
      setState(() {
        _isCameraInitialized = true;
        _statusMessage = '✓ Camera ready. Click Capture & Detect.';
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Error: $e';
      });
    }
  }

  Future<void> _captureAndDetect() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      _showError('Camera not initialized');
      return;
    }

    setState(() {
      _isDetecting = true;
      _statusMessage = '📸 Capturing and detecting...';
    });

    try {
      final image = await _cameraController!.takePicture();
      final bytes = await image.readAsBytes();

      var request = http.MultipartRequest('POST', Uri.parse('$apiUrl/detect'));
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: 'webcam_capture.jpg',
      ));
      request.fields['confidence'] = _confidence.toString();

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var jsonData = json.decode(responseData);

      if (jsonData['success']) {
        setState(() {
          _totalDetections = jsonData['total_detections'];
          _classCount = Map<String, int>.from(jsonData['class_counts']);
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

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
    setState(() {
      _statusMessage = '❌ Error: $message';
    });
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF0A4A8E),
        title: const Text('Live Webcam Detection'),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Camera Preview
            Container(
              color: const Color(0xFF0A4A8E),
              padding: const EdgeInsets.all(10),
              child: Container(
                constraints: const BoxConstraints(
                  minHeight: 400,
                  maxHeight: 600,
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
                    : _isCameraInitialized && _cameraController != null
                        ? ClipRRect(
                            key: const ValueKey('camera_preview'),
                            borderRadius: BorderRadius.circular(8),
                            child: Transform(
                              alignment: Alignment.center,
                              transform: Matrix4.identity()
                                ..scale(_isFlipped ? -1.0 : 1.0, 1.0, 1.0),
                              child: CameraPreview(_cameraController!),
                            ),
                          )
                        : Center(
                            child: Text(
                              _statusMessage,
                              style: const TextStyle(color: Colors.white54, fontSize: 18),
                            ),
                          ),
              ),
            ),
            // Controls
            Container(
              color: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 20),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      ElevatedButton(
                        onPressed: _isDetecting ? null : _captureAndDetect,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF4CAF50),
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                          elevation: 4,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                        ),
                        child: const Text('📸 Capture & Detect', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                      ),
                      const SizedBox(width: 20),
                      ElevatedButton(
                        onPressed: () async {
                          // Clear detection results
                          setState(() {
                            _annotatedImageBase64 = null;
                            _totalDetections = 0;
                            _classCount = {};
                          });
                          
                          // Wait a frame for UI to update
                          await Future.delayed(const Duration(milliseconds: 100));
                          
                          // Update status after clearing
                          setState(() {
                            _statusMessage = '✓ Camera ready. Click Capture & Detect.';
                          });
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFe94560),
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                          elevation: 4,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                        ),
                        child: const Text('🗑️ Clear', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
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
            ),
            // Total Count
            Container(
              height: 60,
              color: const Color(0xFF0A4A8E),
              child: Center(
                child: Text(
                  'TOTAL: $_totalDetections Products',
                  style: const TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFFFFD700),
                  ),
                ),
              ),
            ),
            // Results
            Container(
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
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0xFF084080),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: const Color(0xFFFFD700), width: 2),
                    ),
                    child: Text(
                      _buildResultsText(),
                      style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'Consolas'),
                    ),
                  ),
                ],
              ),
            ),
            // Status Bar
            Container(
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
            ),
          ],
        ),
      ),
    );
  }

  String _buildResultsText() {
    if (_totalDetections == 0) {
      return 'No products detected yet.\nCapture an image to detect products.';
    }

    StringBuffer buffer = StringBuffer();
    buffer.writeln('📦 TOTAL PRODUCTS DETECTED: $_totalDetections\n');
    buffer.writeln('📊 Product Breakdown:');
    _classCount.forEach((className, count) {
      buffer.writeln('   • $className: $count unit(s)');
    });
    return buffer.toString();
  }
}
