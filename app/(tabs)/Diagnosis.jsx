// app/tabs/DiagnosisTab.jsx

import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, Image, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const Diagnosis = () => {
  const [imageUri, setImageUri] = useState(null);
  const [hasPermission, setHasPermission] = useState(null);

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePicture = async (part) => {
    if (hasPermission === null) {
      Alert.alert('Permission status', 'Camera permission is not yet determined.');
      return;
    }

    if (hasPermission === false) {
      Alert.alert('Permission required', 'Camera permission is required to use this feature.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.cancelled) {
      setImageUri(result.uri);
      Alert.alert(`Picture taken for: ${part}`);
    }
  };

  const renderBox = (title) => (
    <View className="bg-white border-2 border-[#40c4c1] rounded-2xl shadow-md p-4 mb-4">
      <Text className="text-lg font-semibold text-center text-[#40c4c1] mb-2">{title}</Text>
      <TouchableOpacity
        className="bg-[#40c4c1] rounded-lg p-3"
        onPress={() => takePicture(title)}
      >
        <Text className="text-black text-center text-lg">Scan {title}</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View className="flex-1 bg-[#d1f6f2] p-4">
      <Text className="text-3xl font-bold text-center mb-10 mt-16">Diagnose Your Pet</Text>

      <View className="flex-row justify-between">
        {/* First row with Eyes and Joints */}
        <View className="flex-1 mr-2">
          {renderBox('Eyes')}
        </View>
        <View className="flex-1 ml-2">
          {renderBox('Joints')}
        </View>
      </View>

      <View className="flex-row justify-between mt-2">
        {/* Second row with Teeth and Skin */}
        <View className="flex-1 mr-2">
          {renderBox('Teeth')}
        </View>
        <View className="flex-1 ml-2">
          {renderBox('Skin')}
        </View>
      </View>

      <View className="bg-white border-2 border-[#40c4c1] rounded-2xl shadow-md p-4 mt-4">
        <Text className="text-lg font-semibold text-center text-[#40c4c1] mb-2">Scan History</Text>
        <Text className="text-center text-gray-600">View your previous scans here.</Text>
      </View>

      {imageUri && (
        <Image
          source={{ uri: imageUri }}
          style={{ width: 300, height: 300, borderRadius: 10, marginTop: 20 }}
        />
      )}
    </View>
  );
};

export default Diagnosis;
