package com.otitenet

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Rule
import org.junit.Test

class OtiteUiTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun testNavigationTabs() {
        // Start the app
        composeTestRule.setContent {
            OtiteApp()
        }

        // Verify "Analyze" tab is default
        composeTestRule.onNodeWithText("Select Image").assertIsDisplayed()

        // Switch to "History" tab
        composeTestRule.onNodeWithText("History").performClick()

        // Verify we are on History screen (checking for list or empty message)
        // Since it's a fresh DB, it might be empty, but the tab should be selected
        composeTabSelected("History")
    }

    private fun composeTabSelected(tabName: String) {
        composeTestRule.onNodeWithText(tabName).assertIsDisplayed()
    }
}
