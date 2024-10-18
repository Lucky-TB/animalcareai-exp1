import { View, Text } from 'react-native';
import { Tabs } from 'expo-router';
import { icons } from '../../constants/icons'; // Adjust the path as necessary
import React from 'react';
import * as Haptics from 'expo-haptics'; // Import Haptics

// TabIcon component to display icons with text
const TabIcon = ({ icon: IconComponent, color, name, focused }) => {
  // Adjust the size of the icon based on whether it is focused
  const iconSize = focused ? 32 : 24;

  // Trigger haptic feedback when focused
  if (focused) {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy); // Trigger haptic feedback
  }

  return (
    <View className="items-center justify-center gap-2 mt-5 mb-4">
      <IconComponent color={color} size={iconSize} />
      <Text className={`${focused ? 'font-semibold ' : 'font-regular'} text-xs`} style={{ color: color, fontSize: focused ? 15 : 12 }}>
        {name}
      </Text>
    </View>
  );
};

const TabsLayout = () => {
  return (
    <>
      <Tabs
        screenOptions={{
          tabBarShowLabel: false,
          tabBarActiveTintColor: '#40c4c1',
          tabBarInactiveTintColor: '#CDCDE0',
          tabBarStyle: {
            backgroundColor: '#161622',
            borderTopWidth: 1,
            borderTopColor: '#232533',
            height: 100,
          },
        }}
      >
        <Tabs.Screen
          name="Home"
          options={{
            title: 'Home',
            headerShown: false,
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.home}
                color={color}
                name="Home"
                focused={focused}
              />
            ),
          }}
        />
        <Tabs.Screen
          name="Diagnosis"
          options={{
            title: 'Diagnosis',
            headerShown: false,
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.diagnosis}
                color={color}
                name="Diagnosis"
                focused={focused}
              />
            ),
          }}
        />
        <Tabs.Screen
          name="Symptoms"
          options={{
            title: 'Symptoms',
            headerShown: false,
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.symptoms}
                color={color}
                name="Symptoms"
                focused={focused}
              />
            ),
          }}
        />
        <Tabs.Screen
          name="PawsyAI"
          options={{
            title: 'PawsyAI',
            headerShown: false,
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.pawsyAI}
                color={color}
                name="PawsyAI"
                focused={focused}
              />
            ),
          }}
        />
      </Tabs>
    </>
  );
};

export default TabsLayout;
