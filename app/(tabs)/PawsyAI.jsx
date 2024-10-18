import React, { useState, useEffect, useRef } from "react";
import * as GoogleGenerativeAI from "@google/generative-ai";
import {
  View,
  Text,
  TextInput,
  FlatList,
  ActivityIndicator,
  TouchableOpacity,
  Image,
  StatusBar as RNStatusBar
} from "react-native";
import * as Speech from "expo-speech";
import { FontAwesome } from "@expo/vector-icons";
import { Entypo } from "@expo/vector-icons";
import FlashMessage, { showMessage } from "react-native-flash-message";
import { StatusBar } from 'expo-status-bar';
import { GEMINI_API_KEY } from '@env';

const AnimalCareAI = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [showStopIcon, setShowStopIcon] = useState(false);

  const flatListRef = useRef(null);

  const API_KEY = GEMINI_API_KEY;  // Use API key from .env file

  useEffect(() => {
    const startChat = async () => {
      try {
        const genAI = new GoogleGenerativeAI.GoogleGenerativeAI(API_KEY);
        const model = genAI.getGenerativeModel({ model: "gemini-pro" });
        const prompt = "Welcome to AnimalCareAI!";
        const result = await model.generateContent(prompt);

        if (result && result.response) {
          const text = result.response.text ? result.response.text() : "No response available.";
          showMessage({
            message: "Welcome to AnimalCareAI 🐾",
            description: text,
            type: "info",
            icon: "info",
            duration: 2000,
          });
          setMessages([{ text, user: false }]);
        } else {
          console.error("Response is undefined or does not have the expected structure", result);
          showMessage({
            message: "Error",
            description: "Failed to start chat. Please try again.",
            type: "danger",
          });
        }
      } catch (error) {
        console.error("Error in startChat:", error);
        showMessage({
          message: "Error",
          description: "An error occurred while starting the chat.",
          type: "danger",
        });
      }
    };
    startChat();
  }, []);

  useEffect(() => {
    flatListRef.current?.scrollToEnd({ animated: true });
  }, [messages]);

  const sendMessage = async () => {
    try {
      setLoading(true);
      const userMessage = { text: userInput, user: true };
      setMessages((prevMessages) => [...prevMessages, userMessage]);

      const genAI = new GoogleGenerativeAI.GoogleGenerativeAI(API_KEY);
      const model = genAI.getGenerativeModel({ model: "gemini-pro" });
      const prompt = `Diagnose animal condition: ${userMessage.text}`;

      const fetchContent = async (retries = 3) => {
        try {
          const result = await model.generateContent(prompt);

          if (result && result.response) {
            const text = result.response.text ? result.response.text() : "No response available.";
            setMessages((prevMessages) => [...prevMessages, { text, user: false }]);
            if (text && !isSpeaking) {
              Speech.speak(text);
              setIsSpeaking(true);
              setShowStopIcon(true);
            }
          } else {
            console.error("Response is undefined or does not have the expected structure", result);
            showMessage({
              message: "Error",
              description: "Failed to generate a response. Please try again.",
              type: "danger",
            });
          }
        } catch (error) {
          if (error.message.includes("503") && retries > 0) {
            console.log(`Retrying... Attempts left: ${retries}`);
            setTimeout(() => fetchContent(retries - 1), 2000);
          } else {
            console.error("Error in fetchContent:", error);
            showMessage({
              message: "Error",
              description: "The service is currently overloaded. Please try again later.",
              type: "danger",
            });
          }
        }
      };

      await fetchContent();

    } catch (error) {
      console.error("Error in sendMessage:", error);
      showMessage({
        message: "Error",
        description: "An error occurred while sending the message.",
        type: "danger",
      });
    } finally {
      setLoading(false);
      setUserInput("");
    }
  };

  const toggleSpeech = () => {
    if (isSpeaking) {
      Speech.stop();
      setIsSpeaking(false);
    } else {
      Speech.speak(messages[messages.length - 1]?.text || "");
      setIsSpeaking(true);
    }
  };

  const ClearMessage = () => {
    setMessages([]);
    setIsSpeaking(false);
  };

  const renderMessage = ({ item }) => (
    <View
      className={`p-3 my-1 rounded-lg shadow-sm ${
        item.user ? "bg-[#f1fcfa] self-end max-w-[80%]" : "self-start max-w-[81%]"
      } flex-row items-start`}
    >
      {!item.user && (
        <Image
          source={require("../../assets/images/icon.png")} // Path to your AI avatar image
          className="w-8 h-8 rounded-full mr-3 ml-[-10]"
        />
      )}
      <Text
        className={`${item.user ? "text-[#08282b]" : "text-[#08282b]"} text-lg`}
      >
        {item.text}
      </Text>
    </View>
  );

  return (
    <View className="flex-1 bg-gradient-to-r from-green-500 via-blue-500 to-indigo-500 p-4 bg-[#d1f6f2]">
      <RNStatusBar barStyle="dark-content" backgroundColor="#d1f6f2" translucent={false} />
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item, index) => index.toString()}
        className="flex-1 mb-3 pt-10 pb-4"
        contentContainerStyle={{ paddingBottom: 40 }}
      />
      <View className="flex-row items-center bg-[#a2ede6] p-3 mb-[-7] w-[100%] rounded-full shadow-md">
        <TouchableOpacity
          className="bg-[#24a8a7] p-3 rounded-full shadow-md"
          onPress={toggleSpeech}
        >
          {isSpeaking ? (
            <FontAwesome name="microphone-slash" size={24} color="white" />
          ) : (
            <FontAwesome name="microphone" size={24} color="white" />
          )}
        </TouchableOpacity>
        <TextInput
          placeholder="Send a message"
          onChangeText={setUserInput}
          value={userInput}
          onSubmitEditing={sendMessage}
          className="flex-1 mx-3 p-3 bg-[#d1f6f2] rounded-full shadow-sm text-gray-800 placeholder-gray-500"
          placeholderTextColor="#aaa"
        />
        {showStopIcon && (
          <TouchableOpacity
            className="bg-red-500 p-3 rounded-full shadow-md"
            onPress={ClearMessage}
          >
            <Entypo name="controller-stop" size={24} color="white" />
          </TouchableOpacity>
        )}
        {loading && <ActivityIndicator size="large" color="#4B5563" className="ml-2" />}
      </View>
    </View>
  );
};

export default AnimalCareAI;