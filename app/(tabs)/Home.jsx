import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Modal } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { AntDesign } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';

const Home = () => {
  const navigation = useNavigation();
  const [modalVisible, setModalVisible] = useState(false);

  return (
    <LinearGradient
      colors={['#d1f6f2', '#d1f6f2']}
      style={{ flex: 1 }}
    >
      <ScrollView contentContainerStyle={{ paddingHorizontal: 20, paddingVertical: 30 }} scrollEnabled={false}>
        {/* Centered Animatable Image */}
        <View style={{ justifyContent: 'center', alignItems: 'center' }}>
          <Animatable.Image
            animation="fadeIn"
            easing="ease-out"
            source={require('/Users/lucky/Desktop/repos/animalcareai/assets/images/animalcare-logo.png')} // Replace with your logo
            style={{ width: 300, height: 300,  marginTop: 15 }}
            resizeMode="contain"
          />
        </View>

        {/* Latest Diagnosis Section (smaller height) */}
        <Animatable.View animation="fadeInUp" duration={800} delay={200} className="mb-4">
          <Text className="text-lg font-bold text-gray-800 mb-2">Latest Diagnosis/Advice</Text>
          <View className="bg-white p-4 rounded-lg shadow-md" style={{ height: 70 }}>
            <Text className="text-gray-600" numberOfLines={1} ellipsizeMode="tail">
              Placeholder diagnosis text goes here, and it will cut off if too long...
            </Text>
          </View>
        </Animatable.View>

        {/* Recent Activity/Updates Section (smaller height) */}
        <Animatable.View animation="fadeInUp" duration={800} delay={400} className="mb-4">
          <Text className="text-lg font-bold text-gray-800 mb-2">Recent Activity/Chats with PawsyAI</Text>
          <View className="bg-white p-4 rounded-lg shadow-md" style={{ height: 70 }}>
            <Text className="text-gray-600" numberOfLines={1} ellipsizeMode="tail">
              Placeholder activity/chat details go here, and it will cut off if too long...
            </Text>
          </View>
        </Animatable.View>

        {/* Buttons Section */}
        <Animatable.View animation="fadeInUp" duration={800} delay={600} className="mb-6">
          <TouchableOpacity
            onPress={() => navigation.navigate('Diagnosis')}
            className="bg-[#40c4c1] rounded-full px-8 py-4 flex-row items-center justify-center mb-4"
            activeOpacity={0.7}
          >
            <Text className="text-black text-lg font-semibold">Start Diagnosis</Text>
            <AntDesign name="arrowright" size={24} color="black" style={{ marginLeft: 10 }} />
          </TouchableOpacity>

          {/* View Recent Activity Button */}
          <TouchableOpacity
            onPress={() => setModalVisible(true)}
            className="bg-[#40c4c1] rounded-full px-8 py-4 flex-row items-center justify-center"
            activeOpacity={0.7}
          >
            <Text className="text-black text-lg font-semibold">View Recent Activity</Text>
            <AntDesign name="eyeo" size={24} color="black" style={{ marginLeft: 10 }} />
          </TouchableOpacity>
        </Animatable.View>
      </ScrollView>

      {/* Modal for Recent Activity */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={{ flex: 1, backgroundColor: 'rgba(0, 0, 0, 0.5)', justifyContent: 'center', alignItems: 'center' }}>
          {/* Increased Modal Height */}
          <View style={{ width: '90%', height: '80%', backgroundColor: '#d1f6f2', borderRadius: 10, padding: 20 }}>
            {/* Modal Content */}
            <Text className="text-3xl justify-center font-bold text-gray-800 mb-4 text-center">Recent Activity</Text>

            {/* Larger Latest Diagnosis/Advice Box in Modal */}
            <View className="mb-8">
              <Text className="text-md text-gray-800 mb-2">Latest Diagnosis/Advice</Text>
              <View className="bg-gray-100 p-4 rounded-lg shadow-md" style={{ maxHeight: 150 }}>
                <ScrollView>
                  <Text className="text-gray-600">
                    Placeholder diagnosis text goes here in the modal. Scroll for more information...
                  </Text>
                </ScrollView>
              </View>
            </View>

            {/* Larger Recent Activity/Chats Box in Modal */}
            <View>
              <Text className="text-md text-gray-800 mb-2">Recent Activity/Chats with PawsyAI</Text>
              <View className="bg-gray-100 p-4 rounded-lg shadow-md" style={{ maxHeight: 150 }}>
                <ScrollView>
                  <Text className="text-gray-600">
                    Placeholder activity/chat details go here in the modal. Scroll for more information...
                  </Text>
                </ScrollView>
              </View>
            </View>

            {/* Close Button */}
            <TouchableOpacity
              onPress={() => setModalVisible(false)}
              className="mt-4 bg-[#40c4c1] rounded-full px-6 py-3 flex-row items-center justify-center"
              activeOpacity={0.7}
            >
              <Text className="text-black text-lg font-semibold">Close</Text>
              <AntDesign className="" name="close" size={24} color="black" style={{ marginLeft: 10 }} />
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </LinearGradient>
  );
};

export default Home;