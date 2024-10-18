// constants/icons.js
import { Entypo, FontAwesome, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import React from 'react';

// Define icon components with default settings
export const icons = {
  home: (props) => <Ionicons name="home" size={24} {...props} />,
  diagnosis: (props) => <MaterialCommunityIcons name="magnify" size={24} {...props} />,
  symptoms: (props) => <FontAwesome name="stethoscope" size={24} {...props} />,
  pawsyAI: (props) => <FontAwesome name="paw" size={24} {...props} />,
};
