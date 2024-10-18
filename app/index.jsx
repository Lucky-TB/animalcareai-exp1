import React, { useState, useRef, useEffect } from 'react';
import { View, Text, ScrollView, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import CustomButton from '../components/CustomButton';
import * as Animatable from 'react-native-animatable';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const OnboardingScreen = () => {
  const [currentPage, setCurrentPage] = useState(0);
  const scrollViewRef = useRef(null);
  const [isOnboardingCompleted, setIsOnboardingCompleted] = useState(false);

  const totalPages = 3;

  useEffect(() => {
    // Check if onboarding has already been completed
    const checkOnboardingStatus = async () => {
      const completed = await AsyncStorage.getItem('onboardingCompleted');
      if (completed === 'true') {
        // If completed, skip onboarding and replace the stack with the home screen
        router.replace('/Home');
      }
    };

    checkOnboardingStatus();
  }, []);

  const handleScroll = (event) => {
    const pageIndex = Math.floor(event.nativeEvent.contentOffset.x / screenWidth);
    setCurrentPage(pageIndex);
  };

  const handleContinue = async () => {
    // Save that the user has seen the onboarding screen
    await AsyncStorage.setItem('onboardingCompleted', 'true');
    setIsOnboardingCompleted(true);
    router.replace('/Home'); // Replace stack to prevent navigating back to onboarding
  };

  if (isOnboardingCompleted) {
    return null; // Don't render anything if onboarding is done
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#d1f6f2' }}>
      <ScrollView
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        ref={scrollViewRef}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        style={{ flexGrow: 1 }}
      >
        {/* First Page */}
        <View style={{ width: screenWidth, justifyContent: 'center', alignItems: 'center', padding: 16 }}>
          <Animatable.Image
            animation="fadeIn"
            easing="ease-out"
            source={require('/Users/lucky/Desktop/repos/animalcareai/assets/images/animalcare-logo.png')} // Replace with your image path
            style={{ width: 300, height: 300, marginBottom: 20 }}
            resizeMode="contain"
          />
          <Text style={{ fontSize: screenHeight * 0.034, color: '#1a202c', fontWeight: 'bold', textAlign: 'center' }}>
            Diagnose Your Pet's Health with <Text style={{ color: '#40c4c1' }}>AnimalCareAI</Text>
          </Text>
          <Text style={{ fontSize: screenHeight * 0.0156, color: '#4a5568', textAlign: 'center', marginTop: 25 }}>
            Advanced AI for Better Animal Care and Health Monitoring.
          </Text>
        </View>

        {/* Page 2: Symptoms Scanner */}
        <View style={{ width: screenWidth, justifyContent: 'center', alignItems: 'center', padding: 16 }}>
          <Animatable.Image
            animation="fadeInUp"
            source={require('/Users/lucky/Desktop/repos/animalcareai/assets/images/cards.png')} // Replace with your image path
            style={{ width: 250, height: 250, marginBottom: 20 }}
            resizeMode="contain"
          />
          <Text style={{ fontSize: screenHeight * 0.034, color: '#1a202c', fontWeight: 'bold', textAlign: 'center' }}>
            Scan Symptoms Easily
          </Text>
          <Text style={{ fontSize: screenHeight * 0.0156, color: '#4a5568', textAlign: 'center', marginTop: 25 }}>
            Use the AI-powered camera feature to analyze symptoms by taking pictures of your pet’s potential health issues.
          </Text>
        </View>

        {/* Page 3: Personalized Reports */}
        <View style={{ width: screenWidth, justifyContent: 'center', alignItems: 'center', padding: 16 }}>
          <Animatable.Image
            animation="fadeInUp"
            source={require('/Users/lucky/Desktop/repos/animalcareai/assets/images/cards.png')} // Replace with your image path
            style={{ width: 250, height: 250, marginBottom: 20 }}
            resizeMode="contain"
          />
          <Text style={{ fontSize: screenHeight * 0.034, color: '#1a202c', fontWeight: 'bold', textAlign: 'center' }}>
            Personalized Health Reports
          </Text>
          <Text style={{ fontSize: screenHeight * 0.0156, color: '#4a5568', textAlign: 'center', marginTop: 25 }}>
            Receive detailed reports on your pet’s health status, including recommended actions and tips for care.
          </Text>
        </View>
      </ScrollView>

      {/* Page Indicator */}
      <View style={{ flexDirection: 'row', justifyContent: 'center', marginVertical: 20 }}>
        {Array.from({ length: totalPages }).map((_, index) => (
          <View
            key={index}
            style={{
              width: currentPage === index ? 20 : 10,
              height: 10,
              borderRadius: 5,
              backgroundColor: currentPage === index ? '#40c4c1' : '#cbd5e0',
              marginHorizontal: 5,
              transition: 'width 0.3s ease-in-out',
            }}
          />
        ))}
      </View>

      {/* Continue Button */}
      {currentPage === totalPages - 1 && (
        <View style={{ alignItems: 'center', marginBottom: 20 }}>
          <CustomButton
            title="Continue to Home"
            handlePress={handleContinue}
            containerStyles={{
              width: screenWidth * 0.7,
              height: 50,
              justifyContent: 'center',
              alignItems: 'center',
              backgroundColor: '#40c4c1',
              borderRadius: 25,
            }}
            textStyle={{
              color: '#fff',
              fontSize: 16,
              fontWeight: 'bold',
            }}
          />
        </View>
      )}
    </SafeAreaView>
  );
};

export default OnboardingScreen;
