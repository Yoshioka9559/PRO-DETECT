import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const ProDetectApp());
}

class ProDetectApp extends StatelessWidget {
  const ProDetectApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ProDetect',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFF0A4A8E),
        scaffoldBackgroundColor: const Color(0xFF1a1a2e),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0A4A8E),
          brightness: Brightness.dark,
        ),
        fontFamily: 'Segoe UI',
      ),
      home: const HomeScreen(),
    );
  }
}
