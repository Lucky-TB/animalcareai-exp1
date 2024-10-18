import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity } from 'react-native';

const Symptoms = () => {
  const [symptoms, setSymptoms] = useState('');
  const [recommendation, setRecommendation] = useState('');

  const checkSymptoms = () => {
    // Mock function for AI diagnosis; replace with real API call
    setRecommendation('Based on the symptoms, your pet might need a vet visit.');
  };

  return (
    <View className="flex-1 justify-center items-center bg-[#d1f6f2] p-6">
      <Text className="text-2xl font-bold text-[#40c4c1] mb-6">Symptom Checker</Text>

      <TextInput
        value={symptoms}
        onChangeText={setSymptoms}
        placeholder="Describe your pet's symptoms"
        className="border border-[#40c4c1] p-4 rounded-lg w-full mb-4 text-base"
        placeholderTextColor="#a1a1a1"
      />

      <TouchableOpacity
        onPress={checkSymptoms}
        className="bg-[#40c4c1] py-3 px-6 rounded-lg w-full items-center shadow-lg"
      >
        <Text className="text-white font-semibold text-lg">Check Symptoms</Text>
      </TouchableOpacity>

      {recommendation && (
        <Text className="mt-6 text-lg text-center text-gray-700">{recommendation}</Text>
      )}
    </View>
  );
};

export default Symptoms;